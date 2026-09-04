import torch
from torch.utils.cpp_extension import load

# 1. Load compiled extension
mod = load(
    name="online_softmax_ext",
    sources=["softmax_online_opt.cu"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)

N = 10 << 20
x = torch.empty(N, device="cuda", dtype=torch.float32).uniform_(0, 100)

# Warmup iterations (crucial to wake up GPU clocks & flush initial launch overhead)
for _ in range(20):
    _ = mod.softmax(x)
    _ = torch.softmax(x, dim=0)

torch.cuda.synchronize()

# ---------------------------------------------------------
# Benchmark PyTorch Native Softmax
# ---------------------------------------------------------
iters = 100
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

torch.cuda.synchronize()
start_event.record()
for _ in range(iters):
    _ = torch.softmax(x, dim=0)
end_event.record()
torch.cuda.synchronize()

torch_ms = start_event.elapsed_time(end_event) / iters

# ---------------------------------------------------------
# Benchmark Custom Online Softmax Kernel
# ---------------------------------------------------------
torch.cuda.synchronize()
start_event.record()
for _ in range(iters):
    _ = mod.softmax(x)
end_event.record()
torch.cuda.synchronize()

custom_ms = start_event.elapsed_time(end_event) / iters

# ---------------------------------------------------------
# Calculate Memory Bandwidth (GB/s)
# ---------------------------------------------------------
# Softmax reads input once (N floats) and writes output once (N floats)
bytes_processed = 2 * N * 4  # 4 bytes per float
torch_gbps = (bytes_processed / 1e9) / (torch_ms / 1000)
custom_gbps = (bytes_processed / 1e9) / (custom_ms / 1000)

print(f"PyTorch Softmax:     {torch_ms:.4f} ms | {torch_gbps:.2f} GB/s")
print(f"Custom Softmax:      {custom_ms:.4f} ms | {custom_gbps:.2f} GB/s")
