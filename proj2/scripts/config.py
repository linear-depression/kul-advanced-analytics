import os
import torch

# ----- Device selection -----
# Apple Silicon (M-series) GPUs are driven through the MPS (Metal) backend, which
# ships inside the standard `torch` wheel — there is no separate "GPU" package on macOS.
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# ----- Training hyperparameters -----
BATCH_SIZE = 64          # Larger batches keep the M-series GPU busy (raise/lower for VRAM).
EPOCHS = 20
LEARNING_RATE = 0.001
TOP_K = 20

# ----- Throughput knobs -----
# Mixed precision runs the conv-heavy ResNet forward pass in float16 on the GPU,
# the single biggest win on MPS/CUDA. Disabled on CPU where it gives nothing.
USE_AMP = DEVICE.type in ("cuda", "mps")

# Parallel DataLoader workers decode/transform images while the GPU trains.
NUM_WORKERS = min(8, os.cpu_count() or 2)

# Pinned host memory only accelerates CUDA host->device copies; it's a no-op on MPS.
PIN_MEMORY = DEVICE.type == "cuda"
