import os
import torch
from torch.utils.data import Dataset
from torchvision.datasets import Flowers102
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

class Flowers102Dataset(Dataset):
    def __init__(self, root, split='train', transform=None, download=True, seed=42):
        """
        Args:
            root (string): Root directory of the dataset.
            split (string): One of 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory.
            seed (int): Random seed for reproducibility.
        """
        self.root = root
        self.split = split
        self.transform = transform
        
        # 1. Load all original splits
        # We use a temporary transform just to load the data structure
        temp_transform = transforms.ToTensor()
        try:
            train_set = Flowers102(root=root, split='train', transform=temp_transform, download=download)
            val_set = Flowers102(root=root, split='val', transform=temp_transform, download=download)
            test_set = Flowers102(root=root, split='test', transform=temp_transform, download=download)
        except Exception as e:
            raise RuntimeError(f"Failed to load Flowers102 dataset: {e}")

        # 2. Merge all data (image paths and labels)
        # Flowers102 uses ._image_files and ._labels internally
        self.image_files = train_set._image_files + val_set._image_files + test_set._image_files
        self.labels = train_set._labels + val_set._labels + test_set._labels
        
        # 3. Stratified Split 70/15/15
        # First split: 70% Train, 30% Temp (Val + Test)
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
        all_indices = np.arange(len(self.labels))
        
        train_idx, temp_idx = next(sss1.split(all_indices, self.labels))
        
        # Second split: Split Temp into 50% Val, 50% Test (which is 15% of total each)
        temp_labels = [self.labels[i] for i in temp_idx]
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        
        val_relative_idx, test_relative_idx = next(sss2.split(temp_idx, temp_labels))
        
        # Map relative indices back to original indices
        val_idx = temp_idx[val_relative_idx]
        test_idx = temp_idx[test_relative_idx]
        
        # Select indices based on requested split
        if self.split == 'train':
            self.indices = train_idx
        elif self.split == 'val':
            self.indices = val_idx
        elif self.split == 'test':
            self.indices = test_idx
        else:
            raise ValueError("Split must be one of 'train', 'val', 'test'")
            
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Map the dataset index to the global index
        global_idx = self.indices[idx]
        
        img_path = self.image_files[global_idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[global_idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms(split='train'):
    """
    Returns the data transformations for the specified split.
    """
    # ImageNet statistics
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            normalize
        ])
    else:
        # Val/Test
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])
