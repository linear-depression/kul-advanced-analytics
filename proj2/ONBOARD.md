# Team Onboarding: Lego Multilabel Vision Project

Welcome to the Lego Minifigure Vision study. This document is designed to help you navigate the codebase, understand the architectural decisions, and set up your environment for review or further experimentation.

## 1. Project Topology
The project is modularized to separate data handling from model logic. This ensures that any component (e.g., the base CNN) can be swapped out without breaking the pipeline.

proj2/
├── data/               <-- Store 'images/' and 'minifigs.json' here
├── outputs/            <-- Automatically generated logs, models, and plots
├── scripts/            <-- The engine room (logic modules)
│   ├── config.py       <-- Hyperparameters and Device (CPU/GPU) settings
│   ├── dataset.py      <-- The PyTorch Dataset & Weighted Sampler logic
│   ├── model.py        <-- ResNet18 architecture + Transfer Learning
│   ├── train.py        <-- The explicit PyTorch training loop
│   ├── evaluate.py     <-- Multilabel metrics and F1 plotting
│   └── interpret.py    <-- Grad-CAM visualization logic
├── main.py             <-- THE ENTRY POINT. Runs the full pipeline.
└── test.py             <-- A smoke-test for data loading/forward passes.

## 2. Environment Setup
We utilize PyTorch for its explicit gradient management. To replicate the environment:

1. Virtual Environment: It is highly recommended to use a venv.
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate   # Windows

2. Dependencies:
    pip install torch torchvision pandas scikit-learn matplotlib seaborn grad-cam tensorboard

## 3. The Logic-Flow (How it works)
When you run main.py, the following sequence occurs:

1. Data Parsing: dataset.py filters for the top-K themes. Because we are in "Choice B" (Multilabel), an image can have multiple '1's in its label vector.
2. Addressing Imbalance: 
    - We use a WeightedRandomSampler to ensure rare minifigs are seen frequently during training. 
    - We calculate a pos_weight tensor to mathematically amplify the loss on minority classes.
3. Transfer Learning: model.py loads a pre-trained ResNet18 and freezes its base. We only train the "head" (the last layer) to map visual features to our 20 Lego themes.
4. Validation & Versioning: Every epoch, the model is tested against a validation set. If the loss is a new all-time low, the model's weights are saved to outputs/saved_models/.
5. Interpretability: After training, interpret.py generates heatmaps using Grad-CAM.

## 4. Key Areas for Review
If you are reviewing this project for a grade or study, focus your attention on:

* scripts/train.py: Observe how the metrics are aggregated manually. We avoid model.fit() to maintain total control over the gradient update steps.
* scripts/dataset.py: Review the LegoDataset class. Notice that we use torch.float32 for labels (required for BCE Loss) and apply ImageNet normalization.
* Loss Function: In main.py, note that we use BCEWithLogitsLoss. It is mathematically superior for this task because it combines the Sigmoid activation and Cross-Entropy loss into one stable numerical step.

## 5. Monitoring
To observe the learning process in real-time, launch TensorBoard from the project root:
    tensorboard --logdir=outputs/logs

You will see dynamic curves for Training vs. Validation Loss. If the Validation loss starts rising while Training loss continues to fall, the model is overfitting.

## 6. Common Troubleshooting
* FileNotFoundError: Ensure you are running your terminal from the proj2/ root directory. Relative paths are set starting from there.
* CUDA/Memory Errors: If the GPU runs out of memory, decrease the BATCH_SIZE in scripts/config.py.
* Grad-CAM Crash: Ensure the grad-cam library is installed via pip. If you change the model architecture, you must update the target_layers in interpret.py.
