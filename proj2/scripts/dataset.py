import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import WeightedRandomSampler
import numpy as np
import scripts.config as config


# ==========================================
# PART 1: Parsing and Splitting the Data
# ==========================================
def prepare_dataframes(json_path, data_dir, top_k=20, sample = None):
    """
    Reads JSON, converts to DataFrame, encodes labels, and splits into Train/Val/Test.
    """
    # 1. Load JSON into a Pandas DataFrame
    df = pd.read_json(json_path)
    if sample: 
        df = df.sample(sample)

    # 2. Keep only the columns we actually need
    # The JSON provided in the assignment uses 'img_local_path' and 'category'
    df = df[['img_local_path', 'themes']].copy()
    
    # 3. Drop rows with missing images or labels
    df = df.dropna()
    
    # 4. Handle Class Imbalance (Crucial Step!)
    # Keep only the top 'k' most frequent categories to prevent the model from failing
    # on categories that only have 1 or 2 images.
    all_themes = [theme for theme_list in df['themes'] for theme in theme_list]
    top_themes = pd.Series(all_themes).value_counts().nlargest(top_k).index.tolist()
    
    # 5. Build absolute file paths
    # 'img_local_path' looks like "images/OLD011.jpg". We prepend the base data directory.
    df['full_path'] = df['img_local_path'].apply(lambda x: os.path.join(data_dir, x))
    
    # 6. Use MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=top_themes)
    labels_matrix = mlb.fit_transform(df['themes']) 
    df['label'] = list(labels_matrix) 
    df['full_path'] = df['img_local_path'].apply(lambda x: os.path.join(data_dir, x))
    
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)
    
    # ==========================================
    # CALCULATE CLASS WEIGHTS (pos_weight for BCE Loss)
    # ==========================================
    train_labels = np.vstack(train_df['label'].values)
    num_positives = train_labels.sum(axis=0)
    num_negatives = len(train_df) - num_positives
    
    # pos_weight = num_negatives / num_positives
    # (We add 1e-5 to prevent dividing by zero)
    pos_weight = num_negatives / (num_positives + 1e-5)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32)

    # ==========================================
    # CALCULATE SAMPLE WEIGHTS (For WeightedRandomSampler)
    # ==========================================
    # Inverse frequency of each class
    class_sample_weights = 1.0 / (num_positives + 1e-5) 
    
    sample_weights = []
    for label_vec in train_labels:
        # Multiply the binary label vector by the class weights.
        # Taking the .max() means the sample's weight is determined by its RAREST theme.
        weight = (label_vec * class_sample_weights).max()
        sample_weights.append(weight)
        
    train_df['sample_weight'] = sample_weights # Save it to the dataframe
    
    # Return the pos_weight_tensor so main.py can give it to the Loss function
    return train_df, val_df, test_df, mlb, pos_weight_tensor

# ==========================================
# PART 2: The PyTorch Dataset Class
# ==========================================
class LegoDataset(Dataset):
    """
    A custom PyTorch Dataset that tells PyTorch how to load ONE image and its label.
    """
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        # PyTorch needs to know exactly how many images are in this dataset
        return len(self.dataframe)

    def __getitem__(self, idx):
        # 1. Get the image path and label for the specific index
        img_path = self.dataframe.loc[idx, 'full_path']
        label = self.dataframe.loc[idx, 'label']
        
        # 2. Open the image using PIL (Python Imaging Library)
        # .convert('RGB') is vital! Some images might be grayscale or RGBA (transparent).
        # Pre-trained models expect exactly 3 color channels (Red, Green, Blue).
        image = Image.open(img_path).convert('RGB')
        
        # 3. Apply mathematical transformations (resizing, tensors, normalization)
        if self.transform:
            image = self.transform(image)
            
        # 4. Convert label to a PyTorch tensor. 
        # dtype=torch.long is strictly required by PyTorch's CrossEntropyLoss
        label_tensor = torch.tensor(label, dtype=torch.float32) 
        
        return image, label_tensor


# ==========================================
# PART 3: The DataLoaders (Transforms & Batching)
# ==========================================
def get_dataloaders(train_df, val_df, test_df, batch_size=32):
    """
    Defines image transformations and wraps datasets in DataLoaders.
    """
    # These magic numbers are the exact mathematical Mean and Standard Deviation 
    # of the ImageNet dataset. Pre-trained models (like ResNet) were trained on 
    # these exact numbers, so we MUST normalize our Lego images to match them.
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Training gets Data Augmentation to prevent overfitting
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),       # Standard size for ResNet/MobileNet
        transforms.RandomHorizontalFlip(),   # Augmentation: Randomly flip images left-to-right
        transforms.RandomRotation(15),       # Augmentation: Randomly tilt images
        transforms.ToTensor(),               # Converts PIL image to a PyTorch Float Tensor (0 to 1)
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # Validation and Test do NOT get augmentations (we want to test on pure images)
    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # Instantiate the Dataset objects
    train_dataset = LegoDataset(train_df, transform=train_transforms)
    val_dataset = LegoDataset(val_df, transform=eval_transforms)
    test_dataset = LegoDataset(test_df, transform=eval_transforms)

    weights = torch.tensor(train_df['sample_weight'].values, dtype=torch.float32)
    
    # Create the sampler. 'replacement=True' allows it to pick minority images multiple times.
    sampler = WeightedRandomSampler(
        weights=weights, 
        num_samples=len(weights), 
        replacement=True
    )

    # Shared throughput settings (see scripts/config.py). persistent_workers keeps the
    # workers (slow to spawn on macOS) alive across epochs instead of re-launching them.
    loader_kwargs = {
        "num_workers": config.NUM_WORKERS,
        "pin_memory": config.PIN_MEMORY,
    }
    if config.NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    # CRITICAL: When using a custom sampler, you MUST set shuffle=False!
    # The sampler handles the shuffling inherently based on the weights.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,   # <-- Pass the sampler here
        shuffle=False,     # <-- Set to False!
        **loader_kwargs,
    )

    # Validation and Test loaders remain completely standard
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader