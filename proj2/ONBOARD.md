# Team Onboarding: Lego Multilabel Vision Project

Welcome to the Lego Minifigure Vision study. This document is designed to help you navigate the codebase, understand the architectural decisions, and set up your environment for review.

## 1. Project Topology
The project is modularized to separate data handling from model logic. This ensures that any component can be swapped out without breaking the pipeline.

~~~text
proj2/
├── data/               <-- Store 'images/' and 'minifigs.json' here
├── outputs/            <-- Automatically generated logs, models, and plots
├── scripts/            <-- The engine room (logic modules)
│   ├── config.py       <-- Hyperparameters and Device settings
│   ├── dataset.py      <-- The PyTorch Dataset & Weighted Sampler
│   ├── model.py        <-- ResNet18 architecture + Transfer Learning
│   ├── train.py        <-- The explicit PyTorch training loop
│   ├── evaluate.py     <-- Multilabel metrics and F1 plotting
│   └── interpret.py    <-- Grad-CAM visualization logic
├── main.py             <-- THE ENTRY POINT. Runs the full pipeline.
└── test.py             <-- A smoke-test for data loading/forward passes.
~~~

## 2. Environment Setup
We utilize PyTorch for its explicit gradient management. To replicate the environment:

**1. Virtual Environment:** It is highly recommended to use a venv.
~~~bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
.\venv\Scripts\activate   # Windows
~~~

**2. Dependencies:**
~~~bash
pip install torch torchvision pandas scikit-learn matplotlib seaborn grad-cam tensorboard
~~~

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
To observe the learning process in real-time, launch TensorBoard from the project root:
~~~bash
tensorboard --logdir=outputs/logs
~~~

## 6. Common Troubleshooting
* **FileNotFoundError:** Ensure your terminal is at the `proj2/` root. 
* **CUDA/Memory Errors:** Decrease the `BATCH_SIZE` in `scripts/config.py`.
* **Grad-CAM Crash:** Ensure the `grad-cam` library is installed.

