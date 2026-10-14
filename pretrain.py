"""Train a paper configuration with python or torchrun; no cluster-specific launcher."""
import argparse
import contextlib
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (FullOptimStateDictConfig, FullStateDictConfig, FullyShardedDataParallel as FSDP,
                                   MixedPrecision, ShardingStrategy, StateDictType)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from model import Config, GPT
from model.data import load_data
from model.fused_cross_entropy import FusedCrossEntropyLoss
from model.model import Block


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data", type=Path, help="JSON manifest of ordered train/validation int32 shards")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/train"))
    p.add_argument("--budget", help="Reference token budget from the config, e.g. 40B (default: largest)")
    p.add_argument("--steps", type=int, help="Override the absolute final optimizer step")
    p.add_argument("--micro-batch-size", type=int)
    p.add_argument("--strategy", choices=("ddp", "fsdp", "hybrid"), default="fsdp")
    p.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--schedule", choices=("wsd", "constant"), default="wsd")
    p.add_argument("--resume", type=Path, help="Full checkpoint from this release trainer")
    p.add_argument("--anneal", action="store_true", help="Start a 1-sqrt tail from --resume without warmup")
    p.add_argument("--save-steps", type=int, nargs="*", default=[])
    p.add_argument("--val-micro-batch-size", type=int, default=1)
    p.add_argument("--verify-data", action="store_true", help="Verify full shard SHA256s before training")
    p.add_argument("--smoke-steps", type=int, help="Run a few updates at the formal LR horizon; save no checkpoint")
    p.add_argument("--dry-run", action="store_true", help="Print model, budget, and parameter count without training")
    p.add_argument("--attention-backend", choices=("sdpa", "flash"))
    p.add_argument("--moe-backend", choices=("torch", "grouped_mm"))
    p.add_argument("--mlp-backend", choices=("torch", "flash"))
    args = p.parse_args()
    if args.anneal and not args.resume:
        p.error("--anneal requires --resume")
    if not args.dry_run and not args.data:
        p.error("--data is required for training")
    if args.val_micro_batch_size <= 0:
        p.error("validation microbatch must be positive")
    if args.smoke_steps is not None and args.smoke_steps <= 0:
        p.error("--smoke-steps must be positive")
    return args


def read_config(args):
    payload = json.loads(args.config.read_text())
    model = dict(payload["model"])
    for key in ("attention_backend", "moe_backend", "mlp_backend"):
        if getattr(args, key) is not None:
            model[key] = getattr(args, key)
    config = Config(**model)
    training = dict(payload["training"])
    if args.micro_batch_size is not None:
        training["micro_batch_size"] = args.micro_batch_size
    budget = args.budget or next(reversed(payload["budgets"]))
    if budget not in payload["budgets"]:
        raise ValueError(f"Unknown budget {budget}; available: {list(payload['budgets'])}")
    endpoint = payload["budgets"][budget]
    steps = endpoint["steps"] if args.steps is None else args.steps
    branch_step = endpoint["branch_step"] if args.steps is None else steps - max(2, math.ceil(steps * training["decay_fraction"]))
    if steps <= 0 or training["micro_batch_size"] <= 0 or training["global_batch_size"] <= 0:
        raise ValueError("Steps and batch sizes must be positive")
    if not 0 <= training["warmup_steps"] <= branch_step < steps:
        raise ValueError("Budget must accommodate warmup and at least two annealing updates")
    if not 0 <= training['min_lr'] <= training['learning_rate'] or training['learning_rate'] <= 0:
        raise ValueError("Require 0 <= minimum LR <= peak LR and peak LR > 0")
    if training["log_every"] <= 0 or training["grad_clip"] <= 0 or training["weight_decay"] < 0:
        raise ValueError("Invalid logging interval, gradient clipping, or weight decay")
    if any(not 0 <= training[k] < 1 for k in ("beta1", "beta2")):
        raise ValueError("AdamW beta values must lie in [0, 1)")
    if any(not 0 < step <= steps for step in args.save_steps):
        raise ValueError("--save-steps must lie within the absolute training horizon")
    return config, training, budget, steps, branch_step


def learning_rate(schedule, step):
    peak, minimum = schedule["peak_lr"], schedule["min_lr"]
    if schedule["kind"] != "anneal":
        if step < schedule["warmup"]:
            return peak * (step + 1) / schedule["warmup"]
        if schedule["kind"] == "constant" or step < schedule["branch"]:
            return peak
    progress = (step - schedule["branch"]) / max(1, schedule["total"] - schedule["branch"] - 1)
    return minimum + (peak - minimum) * (1 - math.sqrt(min(1.0, max(0.0, progress))))


def raw_model(model):
    return model.module if isinstance(model, DDP) else model


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": (state[0], state[1].tolist(), *state[2:]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu())


def checkpoint(model, optimizer, path, metadata, rank, world):
    local_rng = rng_state()
    states = [None] * world
    if world > 1:
        dist.all_gather_object(states, local_rng)
    else:
        states[0] = local_rng
    if isinstance(model, FSDP):
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                 FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                                 FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            weights = model.state_dict()
            optim = FSDP.optim_state_dict(model, optimizer)
    else:
        weights = raw_model(model).state_dict()
        optim = optimizer.state_dict()
    if rank == 0:
        if path.exists():
            raise FileExistsError(f"Checkpoint already exists: {path}")
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(dict(model=weights, optimizer=optim, rng_states=states, **metadata), temporary)
        temporary.replace(path)
    if world > 1:
        dist.barrier()
    restore_rng(local_rng)


def validate(model, stream, micro, device, autocast, rank, world):
    model.eval()
    sums = torch.zeros(2, dtype=torch.float64, device=device)
    loss = FusedCrossEntropyLoss(reduction="sum", inplace_backward=False)
    # All ranks make equally many forwards; tail padding is excluded from NLL.
    stride = world * micro
    with torch.no_grad():
        for base in range(0, stream.windows, stride):
            batch = torch.zeros(micro, stream.length, dtype=torch.long)
            real = max(0, min(micro, stream.windows - base - rank * micro))
            for i in range(real):
                batch[i] = stream.window(base + rank * micro + i)
            batch = batch.to(device)
            with autocast():
                logits = model(batch[:, :-1].contiguous())
            targets = batch[:, 1:].contiguous()
            targets[real:] = -100
            sums[0] += loss(logits, targets).double()
            sums[1] += real * (stream.length - 1)
    if world > 1:
        dist.all_reduce(sums)
    model.train()
    expected = stream.windows * (stream.length - 1)
    if int(sums[1].item()) != expected:
        raise RuntimeError("Validation did not cover every real prediction token exactly once")
    return {"nll": (sums[0] / sums[1]).item(), "prediction_tokens": expected, "windows": stream.windows}


def main(args):
    config, training, budget, total, branch = read_config(args)
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    micro, global_batch = training["micro_batch_size"], training["global_batch_size"]
    if global_batch % (world * micro):
        raise ValueError("Global batch must be divisible by world size times microbatch")
    accumulation = global_batch // (world * micro)
    if args.dry_run:
        with torch.device("meta"):
            model = GPT(config)
        print(json.dumps({"model": asdict(config), "reference_budget": budget, "steps": total,
                          "branch_step": branch, "prediction_tokens": total * global_batch * config.block_size,
                          "parameters": sum(p.numel() for p in model.parameters()),
                          "world_size": world, "gradient_accumulation": accumulation}, indent=2))
        return
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    if world > 1 or "RANK" in os.environ:
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")
    seed = training["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
    if use_cuda:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 1)))
    train, validation = load_data(args.data, config)
    required_steps = total if args.smoke_steps is None else args.smoke_steps
    if train.windows < required_steps * global_batch:
        raise ValueError("Ordered training stream is shorter than the requested budget; wrapping is disabled")
    if args.verify_data:
        error = [None]
        if rank == 0:
            try:
                train.verify_hashes()
                if validation:
                    validation.verify_hashes()
            except (OSError, ValueError) as exc:
                error[0] = str(exc)
        if world > 1:
            dist.broadcast_object_list(error, src=0)
        if error[0] is not None:
            raise ValueError(error[0])
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError("Use a fresh --out-dir, including for resume/annealing branches")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bf16 = args.precision == "bf16"
    autocast = lambda: torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16)
    model = GPT(config).to(device)
    if args.strategy != "ddp" and use_cuda:
        sharding = ShardingStrategy.HYBRID_SHARD if args.strategy == "hybrid" else ShardingStrategy.FULL_SHARD
        if args.strategy == "hybrid" and world <= int(os.environ.get("LOCAL_WORLD_SIZE", world)):
            raise ValueError("hybrid sharding requires multiple nodes; use fsdp on a single node")
        if not dist.is_initialized():
            raise ValueError("CUDA FSDP requires torchrun, including for one GPU")
        model = FSDP(model, auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={Block}),
                     sharding_strategy=sharding, device_id=device, use_orig_params=True,
                     mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
                     if bf16 else None)
    elif world > 1:
        model = DDP(model, device_ids=[device.index] if use_cuda else None,
                    broadcast_buffers=False, gradient_as_bucket_view=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                  betas=(training["beta1"], training["beta2"]),
                                  weight_decay=training["weight_decay"], fused=use_cuda)
    signature = {"model": asdict(config), "training": training, "world_size": world,
                 "precision": args.precision, "strategy": args.strategy,
                 "train": train.identity, "validation": validation.identity if validation else None}
    schedule = {"kind": args.schedule, "total": total, "branch": branch,
                "warmup": training["warmup_steps"], "peak_lr": training["learning_rate"], "min_lr": training["min_lr"]}
    start, last_lr = 0, 0.0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if saved["signature"] != signature:
            raise ValueError("Resume requires the same model, data, optimizer, batch, precision, and GPU topology")
        start, last_lr = saved["step"], saved["last_lr"]
        if saved["consumed_tokens"] != start * global_batch * config.block_size:
            raise ValueError("Checkpoint token cursor is inconsistent")
        if isinstance(model, FSDP):
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                     FullStateDictConfig(offload_to_cpu=True, rank0_only=False)):
                model.load_state_dict(saved["model"])
                optim = FSDP.optim_state_dict_to_load(model, optimizer, saved["optimizer"])
            optimizer.load_state_dict(optim)
        else:
            raw_model(model).load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
        if args.anneal:
            if saved["schedule"]["kind"] != "constant" or total - start < 2:
                raise ValueError("Branch annealing requires a constant-LR checkpoint and two or more remaining steps")
            schedule.update(kind="anneal", branch=start, warmup=0, peak_lr=last_lr)
        else:
            schedule = saved["schedule"]
            if schedule["kind"] == "constant":
                schedule = dict(schedule, total=total)
            elif schedule["total"] != total:
                raise ValueError("Continuation must preserve the original annealing horizon")
        restore_rng(saved["rng_states"][rank])
        del saved
    stop = total if args.smoke_steps is None else min(total, start + args.smoke_steps)
    if train.windows < stop * global_batch:
        raise ValueError("Training data cannot cover the requested absolute progress")
    if not start < stop:
        raise ValueError("Final budget must exceed checkpoint progress")
    resolved = {"model": asdict(config), "training": training, "budget": budget, "steps": total,
                "start_step": start, "gradient_accumulation": accumulation, "world_size": world,
                "schedule": schedule, "data": str(args.data), "resume": str(args.resume) if args.resume else None}
    if rank == 0:
        (args.out_dir / "config.json").write_text(json.dumps(resolved, indent=2) + "\n")
        print(f"{config.name}: world={world}, global_batch={global_batch}, accumulation={accumulation}, steps={total}")
    loss_fn = FusedCrossEntropyLoss()
    optimizer.zero_grad(set_to_none=True)
    begin = time.monotonic()
    for step in range(start, stop):
        last_lr = learning_rate(schedule, step)
        for group in optimizer.param_groups:
            group["lr"] = last_lr
        metrics = torch.zeros(2, device=device)
        for i in range(accumulation):
            first = step * global_batch + (i * world + rank) * micro
            batch = train.batch(first, micro).to(device)
            sync = model.no_sync() if hasattr(model, "no_sync") and i + 1 < accumulation else contextlib.nullcontext()
            with sync, autocast():
                logits, aux = model(batch[:, :-1].contiguous(), return_aux_loss=True)
                nll = loss_fn(logits, batch[:, 1:].contiguous())
                objective = nll + config.router_aux_loss_coef * aux
            (objective / accumulation).backward()
            metrics += torch.stack((nll.detach(), aux.detach())).float() / accumulation
            del logits, aux, nll, objective, batch
        grad_norm = (model.clip_grad_norm_(training["grad_clip"]) if isinstance(model, FSDP)
                     else nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"], error_if_nonfinite=True))
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("Nonfinite gradient norm")
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        completed = step + 1
        if completed % training["log_every"] == 0 or step == start or completed == stop:
            if world > 1:
                dist.all_reduce(metrics); metrics /= world
            if not torch.isfinite(metrics).all():
                raise FloatingPointError("Nonfinite training loss")
            elapsed = time.monotonic() - begin
            record = {"step": completed, "prediction_tokens": completed * global_batch * config.block_size,
                      "nll": metrics[0].item(), "router_aux": metrics[1].item(), "lr": last_lr,
                      "grad_norm": grad_norm.item(), "tokens_per_second": (completed - start) * global_batch * config.block_size / elapsed}
            if rank == 0:
                with (args.out_dir / "metrics.csv").open("a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(record))
                    if f.tell() == 0: writer.writeheader()
                    writer.writerow(record)
                print(f"step={completed}/{total} nll={record['nll']:.6f} lr={last_lr:.6g}", flush=True)
        if args.smoke_steps is None and (completed in args.save_steps or completed == total):
            metadata = {"step": completed, "last_lr": last_lr, "consumed_tokens": completed * global_batch * config.block_size,
                        "signature": signature, "schedule": schedule, "config": asdict(config)}
            name = "final.pt" if completed == total else f"step-{completed:08d}.pt"
            checkpoint(model, optimizer, args.out_dir / name, metadata, rank, world)
    if args.smoke_steps is None and validation:
        report = validate(model, validation, args.val_micro_batch_size, device, autocast, rank, world)
        if rank == 0:
            (args.out_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"validation_nll={report['nll']:.6f}, prediction_tokens={report['prediction_tokens']}")


if __name__ == "__main__":
    try:
        main(parse_args())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
