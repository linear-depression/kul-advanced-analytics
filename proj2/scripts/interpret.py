import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os

def denormalize_image(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = tensor.cpu().numpy().transpose(1, 2, 0)
    image = std * image + mean
    return np.clip(image, 0, 1)

def generate_gradcam(model, test_loader, device, mlb, run_id="latest"):
    print("Generating Comparative Grad-CAM visualization (3 Correct, 3 Incorrect)...")
    
    # 1. Temporarily unfreeze for Grad-CAM
    for param in model.parameters():
        param.requires_grad = True
    model.eval()

    target_layers = [model.layer4[-1]] # Assumes ResNet18/50
    cam = GradCAM(model=model, target_layers=target_layers)

    # 2. Storage for samples
    correct_samples = []
    incorrect_samples = []

    # 3. Search the test_loader until we have enough of both
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs_dev = inputs.to(device)
            outputs = model(inputs_dev)
            
            # Find the top prediction for each image in the batch
            top_preds = torch.argmax(outputs, dim=1)
            
            for i in range(inputs.size(0)):
                pred_idx = top_preds[i].item()
                is_correct = labels[i][pred_idx] == 1 # Is the top-predicted theme actually a true theme?
                
                sample_data = (inputs[i], labels[i], pred_idx)
                
                if is_correct and len(correct_samples) < 3:
                    correct_samples.append(sample_data)
                elif not is_correct and len(incorrect_samples) < 3:
                    incorrect_samples.append(sample_data)
            
            if len(correct_samples) == 3 and len(incorrect_samples) == 3:
                break

    # 4. Create the 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    all_samples = correct_samples + incorrect_samples
    
    for idx, (img_tensor, label_vec, pred_idx) in enumerate(all_samples):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        # Process Grad-CAM
        input_tensor = img_tensor.unsqueeze(0).to(device)
        targets = [ClassifierOutputTarget(pred_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        
        rgb_img = denormalize_image(img_tensor)
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # Get names for display
        pred_name = mlb.classes_[pred_idx]
        true_indices = (label_vec == 1).nonzero(as_tuple=True)[0]
        true_names = ", ".join([mlb.classes_[i] for i in true_indices]) if len(true_indices) > 0 else "None (Filtered)"
        
        # Plotting
        ax.imshow(cam_image)
        ax.axis('off')
        
        status = "CORRECT" if idx < 3 else "INCORRECT"
        color = "green" if idx < 3 else "red"
        
        ax.set_title(f"[{status}]\nTrue: {true_names}\nFocus: {pred_name}", 
                     color=color, fontsize=11, fontweight='bold')

    # 5. Final Formatting and Saving
    plt.tight_layout()
    fig.subplots_adjust(top=0.90, hspace=0.3) # hspace adds room between the rows
    
    save_dir = "proj2/outputs/plots"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/gradcam_comparison_{run_id}.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Comparison Grad-CAM saved to {save_path}")