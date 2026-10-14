"""Ordered, read-only int32 token shards. Prediction windows never wrap."""
import errno
import hashlib
import os
import json
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch


class TokenStream:
    def __init__(self, entries, root, sequence_length, vocab_size):
        self.length = sequence_length + 1
        self.vocab_size = vocab_size
        self.arrays, self.files, self.offsets, self.ends, self.identity = [], [], [], [], []
        total = 0
        for entry in entries:
            entry = {"path": entry} if isinstance(entry, str) else entry
            path = Path(entry.get("path", entry.get("bin", "")))
            if not path.is_absolute():
                path = root / path
            path = path.resolve()
            stat = path.stat()
            offset = entry.get("offset_tokens", 0)
            tokens = entry.get("tokens", stat.st_size // 4 - offset)
            if (stat.st_size % 4 or type(tokens) is not int or tokens <= 0
                    or type(offset) is not int or offset < 0 or offset + tokens > stat.st_size // 4):
                raise ValueError(f"Invalid headerless int32 token range: {path}")
            if offset == 0 and "offset_tokens" not in entry and tokens * 4 != stat.st_size:
                raise ValueError(f"Declared token count differs from file size: {path}")
            try:
                array = np.memmap(path, dtype="<i4", mode="r", offset=offset * 4, shape=(tokens,))
            except OSError as error:
                if error.errno != errno.ENODEV:
                    raise
                array = None  # Some network filesystems do not implement mmap.
            self.arrays.append(array)
            self.files.append(path.open("rb") if array is None else None)
            self.offsets.append(offset)
            total += tokens // self.length
            self.ends.append(total)
            self.identity.append({"path": str(path), "tokens": tokens, "offset_tokens": offset,
                                  "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                  "sha256": entry.get("sha256")})
        self.windows = total
        if total == 0:
            raise ValueError("Token stream has no complete prediction windows")

    def window(self, index):
        if not 0 <= index < self.windows:
            raise IndexError("Token stream exhausted; data wrapping is disabled")
        shard = bisect_right(self.ends, index)
        offset = (index - (self.ends[shard - 1] if shard else 0)) * self.length
        if self.arrays[shard] is None:
            data = os.pread(self.files[shard].fileno(), self.length * 4,
                            (self.offsets[shard] + offset) * 4)
            if len(data) != self.length * 4:
                raise ValueError("Token shard was truncated during reading")
            values = np.frombuffer(data, dtype="<i4").astype(np.int64)
        else:
            values = np.array(self.arrays[shard][offset:offset + self.length], dtype=np.int64)
        if values.min() < 0 or values.max() >= self.vocab_size:
            raise ValueError("Token IDs are outside the configured vocabulary")
        return torch.from_numpy(values)

    def batch(self, start, size):
        return torch.stack([self.window(i) for i in range(start, start + size)])

    def verify_hashes(self):
        for entry in self.identity:
            if entry["sha256"] is None:
                raise ValueError("--verify-data requires SHA256 for every shard")
            digest = hashlib.sha256()
            with Path(entry["path"]).open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 << 20), b""):
                    digest.update(chunk)
            if digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"Data SHA256 mismatch: {entry['path']}")


def load_data(path, config):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get("context_length", config.block_size) != config.block_size:
        raise ValueError("Prepared data context length differs from the model")
    if data.get("ready_marker"):
        ready = path.parent / data["ready_marker"]
        if not ready.is_file() or ready.read_text().strip() != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("Dataset preparation is incomplete or dataset.json changed")
        manifest = path.parent / data["manifest"]
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != data["manifest_sha256"]:
            raise ValueError("Preparation manifest SHA256 mismatch")
    if data.get("train_order", "sequential") != "sequential":
        raise ValueError("Prepare a single fixed, ordered token stream before training")
    if data.get("vocab_size", config.vocab_size) != config.vocab_size:
        raise ValueError("Dataset vocabulary differs from the model")
    train = TokenStream(data["train"], path.parent, config.block_size, config.vocab_size)
    validation = (TokenStream(data["validation"], path.parent, config.block_size, config.vocab_size)
                  if data.get("validation") else None)
    if validation:
        for a in train.identity:
            for b in validation.identity:
                if a['path'] == b['path'] and max(a['offset_tokens'], b['offset_tokens']) < min(
                        a['offset_tokens'] + a['tokens'], b['offset_tokens'] + b['tokens']):
                    raise ValueError("Training and validation token ranges overlap")
    return train, validation
