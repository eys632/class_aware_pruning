#!/usr/bin/env python3
import torch
import time

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPUs:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available.")

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(
        f"cuda:{i}: {p.name}, capability={p.major}.{p.minor}, "
        f"VRAM={p.total_memory/1024**3:.2f} GiB"
    )

device = torch.device("cuda:0")
x = torch.randn(4096, 4096, device=device)
y = torch.randn(4096, 4096, device=device)

# Warm up.
for _ in range(3):
    z = x @ y
torch.cuda.synchronize()

t0 = time.perf_counter()
z = x @ y
torch.cuda.synchronize()
dt = time.perf_counter() - t0

print("Matrix multiplication: OK")
print("shape:", tuple(z.shape))
print(f"single 4096x4096 matmul wall time: {dt*1000:.3f} ms")
