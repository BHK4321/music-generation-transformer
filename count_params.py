import torch
from phase4_benchmark import build_model

# Your model config from train.py defaults
vocab_size = 16000
d_model = 512
num_heads = 8
num_layers = 8
d_ff = 2048
max_seq_len = 1024
dropout = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    d_ff=d_ff,
    max_seq_len=max_seq_len,
    dropout=dropout,
    device=device,
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: {total_params * 4 / (1024**2):.2f} MB (fp32)")

# Breakdown
print("\n=== Parameter Breakdown ===")
for name, param in model.named_parameters():
    params = param.numel()
    print(f"{name}: {params:,}")
