# Team Onboarding: Lego Multilabel Vision Project

Welcome to the Lego Minifigure Vision study. This document is designed to help you navigate the codebase, understand the architectural decisions, and set up your environment for review.

## 1. Project Topology
The project is modularized to separate data handling from model logic. This ensures that any component can be swapped out without breaking the pipeline.

~~~text
proj2/
├── data/               <-- Holds 'images/' and 'minifigs.json' (populated by downloader.py)
├── outputs/            <-- Auto-generated logs, saved_models, and plots
├── notebooks/          <-- EDA.ipynb (exploratory data analysis)
├── scripts/            <-- The engine room (logic modules)
│   ├── config.py       <-- Hyperparameters and Device settings
│   ├── downloader.py   <-- Fetches the image ZIP + minifigs.json into data/
│   ├── dataset.py      <-- The PyTorch Dataset & Weighted Sampler
│   ├── model.py        <-- ResNet18 architecture + Transfer Learning
│   ├── train.py        <-- The explicit PyTorch training loop
│   ├── evaluate.py     <-- Multilabel metrics and F1 plotting
│   └── interpret.py    <-- Grad-CAM visualization logic
├── main.py             <-- THE ENTRY POINT. Runs the full pipeline.
├── pyproject.toml      <-- Project metadata + dependencies (managed by uv)
└── uv.lock             <-- Pinned, reproducible dependency versions
~~~

## 2. Environment Setup
We use [`uv`](https://docs.astral.sh/uv/) to manage the environment and PyTorch for its explicit gradient management. Run every command from the `proj2/` folder.

**1. Install dependencies** (creates an isolated `.venv` from `pyproject.toml` and the pinned `uv.lock`):
~~~bash
uv sync
~~~

**2. Download the dataset** (saves the images and `minifigs.json` into `data/`):
~~~bash
uv run python scripts/downloader.py
~~~

**3. Run the pipeline** (prefix any command with `uv run` to execute it inside the managed environment):
~~~bash
uv run python main.py
~~~

> Don't have `uv`? Install it from https://docs.astral.sh/uv/getting-started/installation/ — or create a Python 3.10+ environment manually and install the dependencies listed in `pyproject.toml`.

## 3. The Logic-Flow
When you run `main.py`, the following sequence occurs:

1. **Data Parsing:** `dataset.py` filters for the top-K themes and binarizes labels.
2. **Addressing Imbalance:** We use a `WeightedRandomSampler` and a `pos_weight` tensor to ensure the model respects minority classes.
3. **Transfer Learning:** `model.py` loads a pre-trained ResNet18. We only train the "head" to map visual features to our 20 themes.
4. **Interpretability:** After training, `interpret.py` generates heatmaps to show where the model is looking.

## 4. Key Areas for Review
* **scripts/train.py**: Observe how metrics are aggregated manually. We avoid `model.fit()` for total control.
* **scripts/dataset.py**: Note the use of `torch.float32` for labels (required for BCE Loss).
* **Loss Function**: In `main.py`, we use `BCEWithLogitsLoss`. It combines Sigmoid and Cross-Entropy into one stable numerical step.

## 5. Monitoring
To observe the learning process in real-time, launch TensorBoard from the `proj2/` folder:
~~~bash
uv run tensorboard --logdir=outputs/logs
~~~

## 6. Common Troubleshooting
* **FileNotFoundError:** Ensure your terminal is in the `proj2/` folder (all paths are relative to it) and that you ran `uv run python scripts/downloader.py` first.
* **GPU / Memory Errors** (`CUDA out of memory` or `MPS backend out of memory`)**:** Lower `BATCH_SIZE` (and/or `NUM_WORKERS`) in `scripts/config.py`.
* **ModuleNotFoundError:** Run commands via `uv run ...` (or `uv sync` first) so they execute inside the managed environment.
* **Numerical instability / NaNs on MPS:** Set `USE_AMP = False` in `scripts/config.py` to fall back to full float32.

## 7. Hardware Acceleration (Apple Silicon & NVIDIA)
`scripts/config.py` auto-selects the device: `mps` on Apple Silicon, `cuda` on NVIDIA, otherwise `cpu`. On a Mac the regular `torch` wheel already ships the Metal/MPS backend, so `uv sync` gives GPU acceleration with no extra package. Verify the GPU is visible:
~~~bash
uv run python -c "import torch; print(torch.backends.mps.is_available())"
~~~

Speed knobs (all in `config.py`):
* `USE_AMP` — float16 mixed-precision forward pass (auto-enabled on MPS/CUDA); the biggest single win.
* `BATCH_SIZE` — larger values keep the GPU busy; reduce on out-of-memory.
* `NUM_WORKERS` — parallel image-loading workers that feed the GPU while it trains.
* `PIN_MEMORY` — auto-enabled on CUDA only (a no-op on MPS).

