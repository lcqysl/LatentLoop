# LatentLoop

<a href="https://arxiv.org/abs/2610.19975"><img src="https://img.shields.io/badge/arXiv-paper-b31b1b?logo=arxiv&logoColor=white" alt="Paper"></a>&nbsp;&nbsp;<a href="https://latentloop-project.github.io/"><img src="https://img.shields.io/badge/%F0%9F%8C%90%20Project-Page-2563eb" alt="Project Page"></a>&nbsp;&nbsp;<a href="https://huggingface.co/yhx12/LatentLoop"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-ffc107" alt="Models"></a>

**LatentLoop: Scaling Looped MoE Models with Asymmetric Compute Allocation**

This repository contains the final LatentLoop architecture and the two baselines used in the paper. It provides nine model configurations and a standalone pretraining entry point. Experiment variants, internal run directories, cluster submission tools, and analysis utilities are omitted.

## Contents

```text
configs/
  1p3b/{moe_base,moe_loop,latentloop}.json
  3b/{moe_base,moe_loop,latentloop}.json
  7b/{moe_base,moe_loop,latentloop}.json
model/              # Transformer, sparse MoE, latent operators, and token reader
pretrain.py          # Single-process and torchrun training
inference.py         # Load released checkpoints and generate text
prepare_data.py      # Offline Parquet/SentencePiece data preparation
requirements.txt
requirements-data.txt
README.md
```

- **MoE Base** executes each layer once.
- **MoE Loop** visits the middle half of the layers twice, sharing its parameters across visits.
- **LatentLoop** visits the same middle half twice, using a pair-mean latent sequence for the first visit and the full sequence for the second. MoE blocks remain shared and attention blocks have separate parameters. A scaled decoder writes the latent increment back to the preserved full sequence.

For zero-based positions, the latent state from tokens `[2j, 2j+1]` can update tokens `2j+1` and `2j+2`. It cannot update token `2j`, which would expose a future token. Latent RoPE uses position `2j+1`. Each layer execution has its own KV cache; incomplete pairs and pending writeback are retained across decoding calls. The decoder stores zero-initialized correction parameters and uses effective coefficients `w = 1 + delta`. Weight decay acts on `delta`, as in the experiments.

## Installation

Python 3.10 or newer is required. The development environment uses PyTorch 2.9.0, CUDA 12.8, and NumPy 2.2.6.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The paper configurations select FlashAttention, its SwiGLU operation, and PyTorch grouped matrix multiplication. Install a FlashAttention build compatible with your PyTorch/CUDA installation separately:

```bash
pip install packaging ninja
pip install flash-attn==2.6.3 --no-build-isolation
```

The experimental environment used FlashAttention 2.6.3. Its optional RMSNorm, rotary, and cross-entropy extensions are supported when installed. PyTorch reference implementations are available for these operations. Grouped GEMMs require CUDA and BF16. Model and expert widths must be divisible by eight; attention head dimensions can differ from model width divided by head count.

For portability or CPU testing, explicitly select the reference operations:

```bash
python pretrain.py --config configs/3b/latentloop.json --dry-run \
  --attention-backend sdpa --moe-backend torch --mlp-backend torch
```

These flags preserve the architecture but change the kernels and throughput. The reference MoE evaluates experts individually and is intended for correctness checks and small experiments.

## Configurations

Each JSON contains `model`, `training`, and `budgets`. All architectures use eight query heads, two KV heads, Top-8 normalized routing, RMSNorm, Q/K normalization, RoPE, SwiGLU experts, and untied input/output embeddings.

| Tier | Model | Stored / executed layers | Width | Head dimension | Experts | Expert width |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1.3B | MoE Base | 12 / 12 | 1024 | 128 | 64 | 512 |
| 1.3B | MoE Loop | 12 / 18 | 864 | 86 | 91 | 432 |
| 1.3B | LatentLoop | 12 / 18 | 944 | 102 | 75 | 472 |
| 3B | MoE Base | 16 / 16 | 1408 | 176 | 64 | 704 |
| 3B | MoE Loop | 16 / 24 | 1184 | 118 | 92 | 592 |
| 3B | LatentLoop | 16 / 24 | 1296 | 140 | 75 | 648 |
| 7B | MoE Base | 20 / 20 | 1856 | 232 | 64 | 928 |
| 7B | MoE Loop | 20 / 30 | 1552 | 154 | 93 | 776 |
| 7B | LatentLoop | 20 / 30 | 1696 | 186 | 76 | 848 |

Tier labels are approximate total parameter counts. The configurations retain the matched experimental widths, head dimensions, expert counts, optimizer steps, and branch positions, even where these values are less convenient for hardware.

| Tier | Reference token budgets |
| --- | --- |
| 1.3B | 5B, 10B, 20B, 40B |
| 3B | 5B, 10B, 20B, 40B, 70B |
| 7B | 5B, 10B, 20B, 40B, 70B, 140B |

Reference budgets use MoE Base's token count. Other architectures use slightly different actual counts to match total FLOPs. The `budgets` object records each endpoint's absolute `steps`, `branch_step`, and actual `prediction_tokens`. No runtime budget fitting is needed.

Training defaults are global batch 256, context 4096, seed 3407, AdamW with betas `(0.9, 0.95)`, weight decay 0.1, gradient clipping 1.0, peak LR 0.001, minimum LR 0.0001, and 500 warmup updates. WSD uses a constant middle phase and a final `1 - sqrt(p)` annealing phase occupying about 20% of the endpoint. Gradient accumulation is inferred from GPU count and microbatch size without changing the global batch.

## Token data

### Offline preparation

The preparation script reads local Parquet files and a local SentencePiece `.model` file. It never downloads data or tokenizer assets. Install its separate CPU dependencies, then run:

```bash
pip install -r requirements-data.txt
python prepare_data.py \
  --input-dir /path/to/fineweb-edu/parquet \
  --tokenizer /path/to/tokenizer.model \
  --output-dir data/fineweb-edu-fixed \
  --workers 8
```

The default recipe uses seed 3407, context length 4096, 256 hash-range buckets, and a 100M prediction-token validation target, rounded up to 100,003,840 prediction tokens. It tokenizes each nonempty document without BOS, appends one EOS, removes exact token-document duplicates, and sorts by the same keyed BLAKE2b shuffle rule as the paper. Validation reserves whole documents at the start of the shuffled unique stream; any extra tokens from its last document remain held out. Training shards contain up to 262,144 complete 4097-ID windows each. A final incomplete window is omitted at each bucket boundary, matching the experimental reader.

The output includes `dataset.json`, `manifest.json`, `preparation.json`, a copy of `tokenizer.model`, and int32 token shards with SHA256s. Source files and tokenizer identity, processing rules, software versions, duplicate counts, reserved validation size, and unused training tails are recorded. Preparation writes to a separate `.partial` directory and publishes the output only after all checks pass. Existing output or partial directories are never overwritten. Only one bucket is sorted in memory at a time; temporary storage holds the tokenized corpus while output is assembled, so use a disk with room for both temporary and final data.

Pass the generated manifest directly to training:

```bash
torchrun --standalone --nproc_per_node=8 pretrain.py \
  --config configs/3b/latentloop.json \
  --data data/fineweb-edu-fixed/dataset.json \
  --budget 40B --out-dir outputs/latentloop-3b-40b
```

For a small preparation check, select a separate small local Parquet directory and reduce `--validation-tokens`. `--context-length`, `--shard-windows`, and `--batch-size` can also be set explicitly. Keep the paper defaults for a matching full dataset. In particular, preparation and training must use the same context length. The full corpus is not prepared automatically by this repository.

### Existing token shards

Prepare headerless little-endian **int32** token shards. The reader treats each shard as consecutive, non-overlapping windows of `context_length + 1` token IDs; each window supplies `context_length` next-token predictions. An incomplete tail is omitted. Shards are read in the manifest order, without reshuffling or wrapping.

Create a JSON manifest, for example `data/dataset.json`:

```json
{
  "vocab_size": 32000,
  "train_order": "sequential",
  "train": [
    {"path": "train-000.bin"},
    {"path": "train-001.bin"}
  ],
  "validation": [
    {"path": "validation.bin"}
  ]
}
```

Paths are relative to the manifest or absolute. Entries may also specify `tokens`, `offset_tokens`, and full-file `sha256`; the internal `bin` spelling is accepted in place of `path`. With `--verify-data`, every shard's SHA256 is checked. Validation is optional, but required to report a final validation NLL. Training and validation token ranges must be disjoint.

The paper uses FineWeb-Edu sample350BT, the 32K TinyLlama tokenizer, no added BOS, and one EOS token (ID 2) per nonempty document. Complete tokenized documents are exactly deduplicated and globally shuffled once with seed 3407. Whole documents are reserved for validation before the remaining documents are concatenated into training shards. The fixed validation panel contains 24,415 windows and 100,003,840 prediction tokens. The reader accepts those original int32 shards without conversion. The dataset and checkpoints are not bundled here; using a different preprocessing pipeline or data order can change the results.

To create a shard from an already prepared list of token IDs:

```python
import numpy as np
np.asarray(token_ids, dtype="<i4").tofile("train-000.bin")
```

## Training

Choose a configuration directly. The largest budget in that config is used by default; `--budget` selects a smaller reference endpoint.

```bash
torchrun --standalone --nproc_per_node=8 pretrain.py \
  --config configs/3b/latentloop.json \
  --data data/dataset.json \
  --budget 40B --out-dir outputs/latentloop-3b-40b
```

Use `configs/3b/moe_base.json` or `configs/3b/moe_loop.json` with the same data to train a baseline. The default strategy is FSDP; `--strategy ddp` runs ordinary distributed training. One-process training uses the same entry point with `python pretrain.py --strategy ddp ...`. CUDA FSDP requires `torchrun`, even with one GPU.

For multiple nodes, use torchrun's standard node and rendezvous arguments:

```bash
torchrun --nnodes=8 --nproc_per_node=8 --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port=29500 pretrain.py \
  --config configs/7b/latentloop.json --data data/dataset.json \
  --strategy hybrid --budget 140B --out-dir outputs/latentloop-7b-140b
```

Set `NODE_RANK` and `MASTER_ADDR` appropriately on each node and launch the command once per node. Hybrid FSDP shards within nodes and replicates across nodes. All ranks share the output directory and manifest. With microbatch four, eight GPUs use eight accumulation steps; 64 GPUs use one. `--micro-batch-size` changes the per-GPU batch while preserving global batch 256.

The output contains `config.json`, training `metrics.csv`, a full `final.pt`, and `validation.json` when validation data is supplied. Validation evaluates every complete window, including the final partial distributed batch, and excludes padding from both loss and token counts. The reported NLL excludes the router auxiliary loss.

A short execution check uses the formal learning-rate horizon, but saves no checkpoint:

```bash
torchrun --standalone --nproc_per_node=8 pretrain.py \
  --config configs/3b/latentloop.json --data data/dataset.json \
  --budget 40B --smoke-steps 20 --out-dir outputs/smoke
```

## Shared scaling mainline and annealing branches

A single endpoint can be trained directly with the default WSD schedule. To reproduce the shared-mainline ladder, train at constant LR through the largest `branch_step`, save the earlier branch positions, then start an annealing branch for each endpoint. For 3B LatentLoop:

```bash
# Save the 5B/10B/20B/40B branch points; the 70B source is final.pt.
torchrun --standalone --nproc_per_node=8 pretrain.py \
  --config configs/3b/latentloop.json --data data/dataset.json \
  --schedule constant --steps 53141 \
  --save-steps 3795 7591 15183 30366 \
  --out-dir outputs/latentloop-3b-mainline

# Resume the 40B branch at absolute step 30366 and anneal to step 37958.
torchrun --standalone --nproc_per_node=8 pretrain.py \
  --config configs/3b/latentloop.json --data data/dataset.json \
  --budget 40B --resume outputs/latentloop-3b-mainline/step-00030366.pt \
  --anneal --out-dir outputs/latentloop-3b-40b
```

Use each configuration's own branch positions: matching FLOPs changes the steps between methods. Budgets and `--steps` are absolute progress, including the resumed prefix. Resume restores model, optimizer, per-rank random states, and the data cursor. Annealing starts at the source checkpoint's last LR and does not repeat warmup. A resumed WSD/annealing run retains its original horizon; a constant mainline can be extended.

Resume requires the same model, data, optimizer settings, global/micro batch, precision, strategy, and GPU count. Always use a fresh output directory. This trainer's full checkpoints have their own training-state format; original internal experiment checkpoints are not supported as resume sources.

## Evaluation

Downstream results in the paper use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), version 0.4.13 ([commit ddd6722](https://github.com/EleutherAI/lm-evaluation-harness/tree/ddd67220430a2470529f25fd5c05a576ca1057a0)). We evaluate `lambada_openai`, `lambada_standard`, `hellaswag`, `piqa`, `winogrande`, `arc_easy`, `arc_challenge`, `sciq`, and `mmlu_continuation`, each in 0-shot and 5-shot settings with few-shot seed 1234 and one evaluation per setting. No chat template or special prefix token is added. We report `acc_norm` for HellaSwag, PIQA, ARC-Easy, ARC-Challenge, and SciQ, and `acc` for the two LAMBADA tasks, WinoGrande, and MMLU continuation; the final average gives equal weight to the nine benchmark scores. For LAMBADA OpenAI at 5-shot, explicitly use the test example pool and exclude the current question. Use FP32 model weights with BF16 forward computation; score continuations with FP32 cross-entropy and FP64 log-probability sums, truncating overlong contexts from the left while retaining the complete scored continuation. To connect these native PyTorch checkpoints to the harness, follow its [custom model guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md).

## Pretrained checkpoints

The largest-budget LatentLoop checkpoint at each scale is available in [yhx12/LatentLoop](https://huggingface.co/yhx12/LatentLoop). Each subfolder contains a separate FP32 model in sharded Safetensors format, its configuration, tokenizer, and training summary.

| Model | Reference token budget | HF subfolder |
| --- | --- | --- |
| LatentLoop 1.3B | 40B | [1.3B-40B](https://huggingface.co/yhx12/LatentLoop/tree/main/1.3B-40B) |
| LatentLoop 3B | 70B | [3B-70B](https://huggingface.co/yhx12/LatentLoop/tree/main/3B-70B) |
| LatentLoop 7B | 140B | [7B-140B](https://huggingface.co/yhx12/LatentLoop/tree/main/7B-140B) |

Install the optional loading dependencies and select a checkpoint directly:

```bash
pip install -r requirements-inference.txt
python inference.py --model yhx12/LatentLoop --subfolder 1.3B-40B \
  --prompt "The capital of France is" --max-new-tokens 32
```

Use `3B-70B` or `7B-140B` for another scale. The loader downloads only the selected checkpoint, validates its tensor index, and retains the original FP32 parameters. CUDA generation uses BF16 autocast, PyTorch SDPA, grouped MoE GEMMs, and PyTorch SwiGLU by default. To select the paper's optional FlashAttention/SwiGLU extensions, pass `--attention-backend flash --mlp-backend flash`. CPU reference inference is available with `--device cpu`. These are base pretrained language models; prompts are encoded without an added BOS or chat template.

The HF model card includes the paper's 0-shot and 5-shot results, original checkpoint hashes, and actual FLOPs-matched token counts. Optimizer and RNG states are omitted from these inference checkpoints.

## Model use

```python
import json
import torch
from model import Config, GPT

checkpoint = torch.load("outputs/latentloop-3b-40b/final.pt", map_location="cpu", weights_only=True)
model = GPT(Config(**checkpoint["config"]))
model.load_state_dict(checkpoint["model"])
model = model.cuda().eval()
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    logits = model(token_ids)  # [batch, sequence, vocabulary]
    model.reset_cache()
    # Cached inputs must start at zero and advance contiguously.
    cached_logits = model(token_ids[:, :1], input_pos=torch.arange(1, device="cuda"))
```

`GPT` also accepts `return_aux_loss=True` and returns `(logits, router_aux_loss)`. Reset the cache before a new sequence, batch, or cache capacity. CPU use requires the reference backend flags or corresponding config values.

## License

Copyright 2026 LatentLoop contributors.

LatentLoop's original code and modifications are licensed under the [Apache License 2.0](LICENSE) (`Apache-2.0`). Third-party source portions retain their original licenses and copyright notices; see [NOTICE](NOTICE).

## Citation

If you find this work useful, please cite:

```bibtex
@article{he2026latentloop,
  title={LatentLoop: Scaling Looped MoE Models with Asymmetric Compute Allocation},
  author={He, Zefeng and Chai, Yunkai and Chen, Xiaoyang and Zhu, Tong and Wu, Zongliang and Luo, Yun and Li, Yafu and Cheng, Yu},
  journal={arXiv preprint arXiv:2610.19975},
  year={2026}
}
```
