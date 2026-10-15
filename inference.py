"""Load released Safetensors checkpoints and generate text with the native model."""
import argparse
import json
from pathlib import Path

import torch

from model import Config, GPT


def load_pretrained(model_path, *, subfolder=None, device="cuda", revision=None,
                    attention_backend="sdpa", moe_backend=None, mlp_backend="torch"):
    """Return (GPT, SentencePieceProcessor) from a local folder or HF repository.

    Parameters remain FP32. Use BF16 autocast on CUDA, as in the paper. The
    default SDPA/torch activation kernels do not require FlashAttention.
    """
    from safetensors.torch import load_file
    import sentencepiece as spm

    if subfolder and (Path(subfolder).is_absolute() or ".." in Path(subfolder).parts):
        raise ValueError("subfolder must be a relative path within the repository")
    root = Path(model_path).expanduser()
    if not root.is_dir():
        from huggingface_hub import snapshot_download
        prefix = subfolder.rstrip("/") + "/" if subfolder else ""
        root = Path(snapshot_download(str(model_path), revision=revision,
                                     allow_patterns=[prefix + "*"]))
    directory = root / subfolder if subfolder else root
    fields = json.loads((directory / "model_config.json").read_text())
    target = torch.device(device)
    fields.update(attention_backend=attention_backend,
                  moe_backend=moe_backend or ("grouped_mm" if target.type == "cuda" else "torch"),
                  mlp_backend=mlp_backend)
    with torch.device("meta"):
        model = GPT(Config(**fields))
    expected = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    weights = index["weight_map"]
    if set(weights) != set(expected):
        raise ValueError("Checkpoint tensor names differ from the configured model")
    seen = set()
    for name in sorted(set(weights.values())):
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("Invalid checkpoint shard filename")
        state = load_file(str(directory / name), device="cpu")
        declared = {key for key, shard in weights.items() if shard == name}
        if set(state) != declared or seen.intersection(state):
            raise ValueError("Missing, duplicated, or unexpected tensors in checkpoint shard")
        for key, value in state.items():
            if tuple(value.shape) != expected[key] or value.dtype != torch.float32:
                raise ValueError(f"Unexpected shape or precision for {key}")
        model.load_state_dict(state, strict=False, assign=True)
        seen.update(state)
        del state
    if seen != set(expected) or any(p.is_meta for p in model.parameters()):
        raise ValueError("Checkpoint did not populate all model parameters")
    model = model.to(target).eval()
    tokenizer = spm.SentencePieceProcessor(model_file=str(directory / "tokenizer/tokenizer.model"))
    if tokenizer.vocab_size() != model.config.vocab_size:
        raise ValueError("Tokenizer vocabulary differs from the model")
    return model, tokenizer


@torch.inference_mode()
def generate(model, token_ids, *, max_new_tokens=64, eos_token_id=2):
    """Greedy continuation with each loop visit's own incremental KV cache."""
    if token_ids.ndim != 2 or token_ids.size(0) != 1 or token_ids.size(1) == 0:
        raise ValueError("Expected a nonempty prompt with batch size one")
    if max_new_tokens <= 0 or token_ids.size(1) + max_new_tokens > model.config.block_size:
        raise ValueError("Prompt plus generated tokens must fit the model's context length")
    tokens = token_ids
    start = 0
    model.reset_cache()
    try:
        for _ in range(max_new_tokens):
            segment = tokens[:, start:]
            positions = torch.arange(start, tokens.size(1), device=tokens.device)
            logits = model(segment, input_pos=positions)
            next_token = logits[:, -1].argmax(-1, keepdim=True)
            start = tokens.size(1)
            tokens = torch.cat((tokens, next_token), dim=1)
            if next_token.item() == eos_token_id:
                break
        return tokens
    finally:
        model.reset_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yhx12/LatentLoop", help="HF repository ID or local folder")
    parser.add_argument("--subfolder", default="1.3B-40B", help="1.3B-40B, 3B-70B, or 7B-140B")
    parser.add_argument("--revision", help="Optional immutable HF commit")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention-backend", choices=("sdpa", "flash"), default="sdpa")
    parser.add_argument("--moe-backend", choices=("torch", "grouped_mm"))
    parser.add_argument("--mlp-backend", choices=("torch", "flash"), default="torch")
    args = parser.parse_args()
    model, tokenizer = load_pretrained(
        args.model, subfolder=args.subfolder, device=args.device, revision=args.revision,
        attention_backend=args.attention_backend, moe_backend=args.moe_backend,
        mlp_backend=args.mlp_backend)
    prompt = tokenizer.encode(args.prompt, out_type=int)  # No added BOS or chat template.
    tokens = torch.tensor([prompt], dtype=torch.long, device=args.device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.device(args.device).type == "cuda"):
        output = generate(model, tokens, max_new_tokens=args.max_new_tokens,
                          eos_token_id=tokenizer.eos_id())
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
