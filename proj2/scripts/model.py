import torchvision.models as models
import torch.nn as nn

def build_model(num_classes):
    # 1. Download the pre-trained model (e.g., ResNet18)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # 2. Freeze the base layers (so we don't destroy pre-trained weights)
    for param in model.parameters():
        param.requires_grad = False
        
    # 3. Replace the 'head' (the final MLP layer)
    # ResNet's final layer is named 'fc'. We replace it with a new, untrained Dense (Linear) layer.
    # This new layer automatically has requires_grad=True
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model