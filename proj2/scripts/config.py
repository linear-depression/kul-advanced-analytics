import torch
# Automatically use NVIDIA GPU or Apple Silicon if available, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
TOP_K = 20
