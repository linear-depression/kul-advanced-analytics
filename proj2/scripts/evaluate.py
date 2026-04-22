import torch
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(trained_model, test_loader, config_device, mlb, run_id="latest"):
    trained_model.eval()

    all_predicted = []
    all_true_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(config_device), labels.to(config_device)
            outputs = trained_model(inputs)

            # Multilabel logic
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).float()

            all_predicted.append(predicted.cpu().numpy())
            all_true_labels.append(labels.cpu().numpy())

    # Stack batches into single large numpy arrays
    all_predicted = np.vstack(all_predicted)
    all_true_labels = np.vstack(all_true_labels)

    # 1. Classification Report (Works perfectly for Multilabel out of the box!)
    target_names = mlb.classes_
    print("\n--- Multilabel Classification Report ---")
    report = metrics.classification_report(all_true_labels, all_predicted, target_names=target_names)
    print(report)

    # 2. Plot F1-Scores instead of a Confusion Matrix
    report_dict = metrics.classification_report(all_true_labels, all_predicted, target_names=target_names, output_dict=True)
    
    # Extract F1 scores for just the themes (ignoring 'micro avg', 'macro avg', etc.)
    f1_scores = [report_dict[theme]['f1-score'] for theme in target_names]

    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(target_names), y=f1_scores, palette="viridis")
    plt.title('F1-Score per Theme')
    plt.ylabel('F1-Score')
    plt.xlabel('Lego Themes')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    save_dir = "proj2/outputs/plots"
    os.makedirs(save_dir, exist_ok=True) 
    save_path = f"{save_dir}/f1_scores_{run_id}.png"
    plt.savefig(save_path) 
    print(f"F1-Score chart saved to {save_path}")