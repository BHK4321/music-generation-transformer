import argparse
import os
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp

from phase4_benchmark import build_model


@dataclass
class TrainConfig:
    token_file: str
    output_dir: str
    seq_len: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    max_grad_norm: float
    train_split: float
    seed: int
    # Model config
    vocab_size: int
    d_model: int
    num_heads: int
    num_layers: int
    d_ff: int
    max_seq_len: int
    dropout: float
    num_workers: int
    # Resume from checkpoint
    resume_from: str | None = None


class NextTokenDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, seq_len: int):
        if tokens.dim() != 1:
            raise ValueError("tokens must be a 1D LongTensor of token IDs")
        if tokens.numel() <= seq_len:
            raise ValueError("Not enough tokens for the chosen seq_len")
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self) -> int:
        # Each sample consumes a contiguous block of seq_len + 1 tokens.
        # Blocks do not overlap; any remainder at the end is dropped.
        return (self.tokens.numel() - 1) // self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]
        if chunk.numel() < self.seq_len + 1:
            raise IndexError("Requested block is incomplete")
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def load_tokens(path: str) -> torch.Tensor:
    token_path = Path(path)
    if not token_path.exists():
        raise FileNotFoundError(f"Token file not found: {token_path}")

    if token_path.suffix == ".pt":
        tokens = torch.load(token_path, map_location="cpu")
        if isinstance(tokens, dict) and "tokens" in tokens:
            tokens = tokens["tokens"]
        if not isinstance(tokens, torch.Tensor):
            raise ValueError(".pt file must contain a torch.Tensor or a dict with key 'tokens'")
        tokens = tokens.long().flatten()
        return tokens

    if token_path.suffix == ".txt":
        raw = token_path.read_text(encoding="utf-8").strip().split()
        tokens = torch.tensor([int(x) for x in raw], dtype=torch.long)
        return tokens

    raise ValueError("Unsupported token file format. Use .pt or .txt")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed() -> Tuple[bool, int, int]:
    distributed = is_distributed()
    if not distributed:
        return False, 0, 1

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return True, local_rank, get_world_size()


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _ddp_worker(rank: int, world_size: int, config: TrainConfig, master_port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    train(config)


def launch_ddp_training(config: TrainConfig, world_size: int | None = None, master_port: int = 29501) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("DDP launcher requires CUDA GPUs")

    available_gpus = torch.cuda.device_count()
    if available_gpus < 2:
        raise RuntimeError(f"DDP launcher needs at least 2 GPUs, but found {available_gpus}")

    actual_world_size = world_size or min(available_gpus, 2)
    if actual_world_size < 2:
        raise RuntimeError("DDP launcher needs at least 2 processes")

    mp.spawn(
        _ddp_worker,
        args=(actual_world_size, config, master_port),
        nprocs=actual_world_size,
        join=True,
    )


def split_tokens(tokens: torch.Tensor, train_split: float) -> Tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split must be between 0 and 1")
    split_idx = int(tokens.numel() * train_split)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]
    if train_tokens.numel() < 2 or val_tokens.numel() < 2:
        raise ValueError("Train/val split is too small; provide more tokens")
    return train_tokens, val_tokens


def build_scheduler(optimizer: AdamW, warmup_steps: int, total_steps: int):
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device, criterion: nn.Module, distributed: bool) -> float:
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_batches = torch.tensor(0.0, device=device)

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        log_probs = model(x)
        loss = criterion(log_probs.view(-1, log_probs.size(-1)), y.view(-1))
        total_loss += loss.detach()
        total_batches += 1.0

    if distributed:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_batches, op=dist.ReduceOp.SUM)

    denom = total_batches.clamp_min(1.0)
    return (total_loss / denom).item()


def train(config: TrainConfig) -> None:
    distributed, local_rank, world_size = setup_distributed()

    rank = get_rank()
    set_seed(config.seed + rank)

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP was requested but CUDA is not available")
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(config.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    tokens = load_tokens(config.token_file)
    train_tokens, val_tokens = split_tokens(tokens, config.train_split)

    train_dataset = NextTokenDataset(train_tokens, config.seq_len)
    val_dataset = NextTokenDataset(val_tokens, config.seq_len)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        device=device,
    )

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.NLLLoss()

    total_steps = config.epochs * max(1, len(train_loader))
    scheduler = build_scheduler(optimizer, config.warmup_steps, total_steps)

    best_val_loss = float("inf")
    global_step = 0
    start_epoch = 1

    # Load checkpoint if resuming
    if config.resume_from and is_main_process():
        resume_path = Path(config.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("val_loss", float("inf"))
        
        raw_model = model.module if isinstance(model, DDP) else model
        raw_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if is_main_process():
            print(f"Loaded checkpoint from {resume_path}", flush=True)
            print(f"Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}", flush=True)

    if is_main_process():
        print("=== Training Setup ===", flush=True)
        print(f"Distributed: {distributed}", flush=True)
        print(f"World size: {world_size}", flush=True)
        print(f"Device: {device}", flush=True)
        print(f"Per-GPU batch size: {config.batch_size}", flush=True)
        print(f"Global batch size: {config.batch_size * world_size}", flush=True)
        print(f"Train tokens: {train_tokens.numel()}", flush=True)
        print(f"Val tokens: {val_tokens.numel()}", flush=True)
        print(f"Train batches/epoch (per rank): {len(train_loader)}", flush=True)
        print(f"Val batches/epoch (per rank): {len(val_loader)}", flush=True)
        print(f"Model config: layers={config.num_layers}, d_model={config.d_model}, heads={config.num_heads}, d_ff={config.d_ff}", flush=True)
        print(flush=True)

    for epoch in range(start_epoch, config.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            log_probs = model(x)
            loss = criterion(log_probs.view(-1, log_probs.size(-1)), y.view(-1))
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1

            if is_main_process() and global_step % 200 == 0:
                avg = running_loss / 200.0
                print(f"Step {global_step}: train_loss={avg:.4f}, lr={scheduler.get_last_lr()[0]:.6e}", flush=True)
                running_loss = 0.0

        if is_main_process():
            print(f"Epoch {epoch}: finished training batches, starting train-set evaluation...", flush=True)

        train_loss = evaluate(model, train_loader, device, criterion, distributed)

        if is_main_process():
            print(f"Epoch {epoch}: train-set evaluation done, starting val-set evaluation...", flush=True)

        val_loss = evaluate(model, val_loader, device, criterion, distributed)
        train_ppl = math.exp(min(train_loss, 20.0))
        val_ppl = math.exp(min(val_loss, 20.0))

        if is_main_process():
            print(f"Epoch {epoch}/{config.epochs}: train_loss={train_loss:.4f}, train_ppl={train_ppl:.2f}, val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}", flush=True)

        if is_main_process():
            raw_model = model.module if isinstance(model, DDP) else model
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "config": vars(config),
            }

            last_ckpt = output_dir / "last.pt"
            torch.save(checkpoint, last_ckpt)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt = output_dir / "best.pt"
                torch.save(checkpoint, best_ckpt)
                print(f"Saved new best checkpoint to {best_ckpt}", flush=True)

    cleanup_distributed()



def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train GPTmini with next-token prediction using build_model utility")

    parser.add_argument("--token-file", type=str, required=True, help="Path to token IDs (.pt or .txt)")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume training from")

    args = parser.parse_args()
    return TrainConfig(
        token_file=args.token_file,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        train_split=args.train_split,
        seed=args.seed,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        num_workers=args.num_workers,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    config = parse_args()
    train(config)
