"""Offline Parquet -> deduplicated, fixed-order int32 token shards.

Only local input files and a local SentencePiece model are accepted. Hash-range
buckets bound memory use and preserve the paper's document ordering and tails.
"""
import argparse
import contextlib
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import resource
import shutil
import struct

import numpy as np
import pyarrow.parquet as pq
import sentencepiece as sp

BUCKETS = 256
RECORD = struct.Struct("<16s16sQIIQ")
INDEX = np.dtype([("key", "S16"), ("digest", "S16"), ("offset", "<u8"),
                  ("length", "<u4"), ("source", "<u4"), ("row", "<u8")])


def token_count(value):
    text = str(value).strip().upper()
    multiplier = {"K": 1000, "M": 1000000, "B": 1000000000}.get(text[-1:], 1)
    try:
        number = int(text[:-1] if multiplier != 1 else text) * multiplier
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use a nonnegative integer, e.g. 100M") from error
    if number < 0:
        raise argparse.ArgumentTypeError("Token count must be nonnegative")
    return number


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class ShardWriter:
    def __init__(self, root, prefix, max_tokens, window):
        self.root, self.prefix, self.limit, self.window = root, prefix, max_tokens, window
        self.stream = None
        self.entries = []
        self.total = self.current = 0

    def write(self, tokens):
        while len(tokens):
            if self.stream is None:
                self.path = self.root / f"{self.prefix}-{len(self.entries):04d}.bin"
                self.stream = self.path.open("xb", buffering=1 << 20)
                self.digest = hashlib.sha256()
                self.current = 0
            size = min(len(tokens), self.limit - self.current)
            body = tokens[:size].astype("<i4", copy=False).tobytes()
            self.stream.write(body)
            self.digest.update(body)
            self.current += size
            self.total += size
            tokens = tokens[size:]
            if self.current == self.limit:
                self.close_file()

    def close_file(self):
        if self.stream is not None:
            self.stream.close()
            self.entries.append({"path": self.path.name, "tokens": self.current,
                                 "prediction_tokens": (self.current // self.window) * (self.window - 1),
                                 "sha256": self.digest.hexdigest()})
            self.stream = None

    def close(self):
        self.close_file()
        return self.entries


def tokenize(args, files, tokenizer, scratch, shuffle_key):
    offsets = [0] * BUCKETS
    sources, total_documents, total_tokens, empty = [], 0, 0, 0
    # Open buckets lazily, and close all descriptors even if tokenization fails.
    with contextlib.ExitStack() as stack:
        streams = {}
        for source, path in enumerate(files):
            before = path.stat()
            parquet = pq.ParquetFile(path)
            row = 0
            for batch in parquet.iter_batches(batch_size=args.batch_size,
                                              columns=[args.text_column], use_threads=False):
                texts = batch.column(0).to_pylist()
                selected = [(row + i, text) for i, text in enumerate(texts)
                            if isinstance(text, str) and text.strip()]
                encoded = tokenizer.encode([text for _, text in selected], out_type=int,
                                           num_threads=args.workers) if selected else []
                empty += len(texts) - len(selected)
                for (position, _), values in zip(selected, encoded):
                    if not values:
                        empty += 1
                        continue
                    ids = np.asarray(values, dtype="<i4")
                    eos = tokenizer.eos_id()
                    if ids.min() < 0 or ids.max() >= tokenizer.vocab_size() or np.any(ids == eos):
                        raise ValueError(f"Invalid token IDs or embedded EOS at {path.name}:{position}")
                    body = np.empty(len(ids) + 1, dtype="<u2")
                    body[:-1], body[-1] = ids, eos
                    data = body.tobytes()
                    digest = hashlib.blake2b(data, digest_size=16).digest()
                    key = hashlib.blake2b(digest, digest_size=16, key=shuffle_key).digest()
                    bucket = key[0]
                    if bucket not in streams:
                        streams[bucket] = (
                            stack.enter_context((scratch / f"{bucket:03d}.bin").open("xb", buffering=1 << 16)),
                            stack.enter_context((scratch / f"{bucket:03d}.idx").open("xb", buffering=1 << 16)),
                        )
                    data_stream, index_stream = streams[bucket]
                    data_stream.write(data)
                    index_stream.write(RECORD.pack(key, digest, offsets[bucket], len(body), source, position))
                    offsets[bucket] += len(body)
                    total_documents += 1
                    total_tokens += len(body)
                row += len(texts)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"Source changed during preparation: {path}")
            sources.append({"path": str(path.relative_to(args.input_dir)), "bytes": after.st_size,
                            "rows": row, "sha256": file_sha256(path)})
            print(f"tokenized {source + 1}/{len(files)} files; documents={total_documents:,}", flush=True)
    if total_documents + empty != sum(source["rows"] for source in sources):
        raise RuntimeError("Source row accounting mismatch")
    return sources, {"input_documents": total_documents, "input_token_ids": total_tokens,
                     "empty_documents": empty}


def assemble(args, scratch, destination, validation_ids, shuffle_key):
    window = args.context_length + 1
    shard_ids = args.shard_windows * window
    train_entries = []
    # Reserve whole documents; keep even the held-out final document's excess IDs.
    validation = ShardWriter(destination, "validation", 2**63 - 1, window)
    unique = duplicates = unique_ids = duplicate_ids = train_docs = val_docs = 0
    document_digests = {"train": hashlib.sha256(), "validation": hashlib.sha256()}
    try:
        for bucket in range(BUCKETS):
            index_path = scratch / f"{bucket:03d}.idx"
            if not index_path.exists():
                continue
            if index_path.stat().st_size % INDEX.itemsize:
                raise ValueError("Truncated temporary document index")
            refs = np.fromfile(index_path, dtype=INDEX)
            if not len(refs):
                raise ValueError("Empty temporary document index")
            # Only this bucket's payload and index are in RAM, never the full corpus.
            payload = np.fromfile(scratch / f"{bucket:03d}.bin", dtype="<u2")
            order = np.argsort(refs, order=["key", "digest", "source", "row"], kind="stable")
            train = ShardWriter(destination, f"train-{bucket:03d}", shard_ids, window)
            previous_key = previous_digest = previous_body = None
            try:
                for index in order:
                    ref = refs[index]
                    key = ref["key"].tobytes().ljust(16, b"\0")
                    digest = ref["digest"].tobytes().ljust(16, b"\0")
                    offset, length = int(ref["offset"]), int(ref["length"])
                    body = payload[offset:offset + length]
                    if length == 0 or len(body) != length or hashlib.blake2b(body.tobytes(), digest_size=16).digest() != digest:
                        raise ValueError("Temporary token-document identity mismatch")
                    if hashlib.blake2b(digest, digest_size=16, key=shuffle_key).digest() != key:
                        raise ValueError("Temporary shuffle key mismatch")
                    if previous_key == key:
                        if previous_digest != digest or not np.array_equal(previous_body, body):
                            raise ValueError("Hash collision: refusing ambiguous exact deduplication")
                        duplicates += 1
                        duplicate_ids += length
                        continue
                    previous_key, previous_digest, previous_body = key, digest, body
                    unique += 1
                    unique_ids += length
                    split = "validation" if validation.total < validation_ids else "train"
                    document_digests[split].update(digest)
                    if split == "validation":
                        validation.write(body)
                        val_docs += 1
                    else:
                        train.write(body)
                        train_docs += 1
            finally:
                train.close()
            train_entries.extend(train.entries)
            print(f"sorted bucket {bucket:03d}; unique={unique:,}, duplicates={duplicates:,}", flush=True)
            del payload, refs, order, previous_body, body, ref
    finally:
        validation.close()
    if validation.total < validation_ids or not train_entries:
        raise ValueError("Input is too small for the requested validation split and training stream")
    val_entries = validation.entries
    if val_entries:
        if len(val_entries) != 1:
            raise RuntimeError("Validation must be one contiguous reserved stream")
        val_entries[0].update(tokens=validation_ids, offset_tokens=0,
                              prediction_tokens=(validation_ids // window) * args.context_length)
    if not any(entry["prediction_tokens"] for entry in train_entries):
        raise ValueError("Training stream has no complete prediction windows")
    return train_entries, val_entries, {"unique_documents": unique, "unique_token_ids": unique_ids,
        "duplicate_documents": duplicates, "duplicate_token_ids": duplicate_ids,
        "train_documents": train_docs, "validation_documents": val_docs,
        "validation_reserved_token_ids": validation.total,
        "ordered_document_digest_sha256": {k: v.hexdigest() for k, v in document_digests.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Local directory of Parquet files (recursive)")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Local SentencePiece .model file")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--validation-tokens", type=token_count, default=100000000, help="Prediction tokens; rounded up to complete windows")
    parser.add_argument("--shard-windows", type=int, default=262144, help="Complete windows per full training shard")
    parser.add_argument("--workers", type=int, default=8, help="SentencePiece tokenization threads")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if min(args.context_length, args.shard_windows, args.workers, args.batch_size) <= 0 or not 0 <= args.seed < 2**64:
        parser.error("Sizes and workers must be positive; seed must be an unsigned 64-bit integer")
    args.input_dir = args.input_dir.resolve()
    args.tokenizer = args.tokenizer.resolve()
    args.output_dir = args.output_dir.resolve()
    files = sorted(args.input_dir.rglob("*.parquet"))
    if not args.input_dir.is_dir() or not files:
        parser.error("--input-dir must contain local Parquet files")
    if not args.tokenizer.is_file():
        parser.error("--tokenizer must be a local SentencePiece model file")
    partial = args.output_dir.with_name(args.output_dir.name + ".partial")
    if args.output_dir.exists() or partial.exists():
        parser.error("Use a new output directory; existing outputs and partial runs are never overwritten")
    tokenizer = sp.SentencePieceProcessor(model_file=str(args.tokenizer))
    if not 0 < tokenizer.vocab_size() <= 65536 or not 0 <= tokenizer.eos_id() < tokenizer.vocab_size():
        parser.error("Tokenizer must have an EOS ID and a vocabulary of at most 65536 tokens")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    needed = 2 * BUCKETS + 32
    if soft < needed:
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(needed, hard) if hard != resource.RLIM_INFINITY else needed, hard))
        if resource.getrlimit(resource.RLIMIT_NOFILE)[0] < needed:
            parser.error(f"File descriptor limit must be at least {needed}")
    window = args.context_length + 1
    validation_windows = (args.validation_tokens + args.context_length - 1) // args.context_length
    validation_ids = validation_windows * window
    shuffle_key = f"LatentLoop-350BT-document-shuffle-seed-{args.seed}-v1".encode()
    partial.mkdir(parents=True)
    scratch = partial / "_temporary"
    scratch.mkdir()
    settings = {"code_sha256": file_sha256(Path(__file__)),
                "packages": {name: version(name) for name in ("numpy", "pyarrow", "sentencepiece")},
                "seed": args.seed, "shuffle_key": shuffle_key.decode(), "buckets": BUCKETS,
                "context_length": args.context_length, "shard_windows": args.shard_windows,
                "requested_validation_prediction_tokens": args.validation_tokens,
                "validation_prediction_tokens": validation_windows * args.context_length,
                "text_column": args.text_column, "tokenizer_sha256": file_sha256(args.tokenizer),
                "vocab_size": tokenizer.vocab_size(), "eos_id": tokenizer.eos_id(),
                "format": "little-endian int32; no BOS; one EOS per nonempty tokenized document",
                "deduplication": "exact token documents, with equality checked on duplicate hashes",
                "shuffle": "ascending keyed BLAKE2b-128 of BLAKE2b-128(little-endian uint16 token document)",
                "validation_selection": "whole documents from the start of the shuffled unique stream",
                "training_window_policy": "non-overlapping context+1 ID windows; drop only each bucket's final incomplete window"}
    write_json(partial / "preparation.json", settings)
    sources, counts = tokenize(args, files, tokenizer, scratch, shuffle_key)
    train, validation, assembled = assemble(args, scratch, partial, validation_ids, shuffle_key)
    if (assembled["unique_documents"] + assembled["duplicate_documents"] != counts["input_documents"]
            or assembled["unique_token_ids"] + assembled["duplicate_token_ids"] != counts["input_token_ids"]):
        raise RuntimeError("Document/token conservation check failed")
    if file_sha256(args.tokenizer) != settings["tokenizer_sha256"]:
        raise ValueError("Tokenizer changed during preparation")
    shutil.copyfile(args.tokenizer, partial / "tokenizer.model")
    manifest = {"status": "complete", "preparation": settings, "sources": sources,
                **counts, **assembled, "train_prediction_tokens": sum(e["prediction_tokens"] for e in train),
                "train_unused_tail_token_ids": sum(e["tokens"] % window for e in train)}
    write_json(partial / "manifest.json", manifest)
    dataset = {"vocab_size": tokenizer.vocab_size(), "context_length": args.context_length,
               "train_order": "sequential", "tokenizer": "tokenizer.model", "ready_marker": "DATASET_READY",
               "train": train, "validation": validation, "manifest": "manifest.json",
               "manifest_sha256": file_sha256(partial / "manifest.json")}
    write_json(partial / "dataset.json", dataset)
    shutil.rmtree(scratch)
    (partial / "DATASET_READY").write_text(file_sha256(partial / "dataset.json") + "\n")
    partial.rename(args.output_dir)
    print(f"Ready: {args.output_dir / 'dataset.json'}; training prediction tokens={manifest['train_prediction_tokens']:,}", flush=True)


if __name__ == "__main__":
    main()
