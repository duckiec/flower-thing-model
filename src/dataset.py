import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import os
from typing import Tuple, Optional

class FlowerDataset(Dataset):
    def __init__(self, data_dir: str, transform: Optional[A.Compose] = None, train: bool = True):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        
        # Get all classes (flower types)
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._get_samples()
        
        # Calculate class weights for balanced sampling
        self._calculate_class_weights()
        print(f"Found {len(self.classes)} classes and {len(self.samples)} images")
    
    def _calculate_class_weights(self):
        """Calculate weights for balanced sampling"""
        class_counts = {}
        for _, label in self.samples:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        total_samples = len(self.samples)
        self.class_weights = {cls: total_samples / (len(class_counts) * count)
                            for cls, count in class_counts.items()}
        
        self.sample_weights = [self.class_weights[label] for _, label in self.samples]
    
    def _get_samples(self):
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    samples.append((
                        os.path.join(class_dir, img_name),
                        self.class_to_idx[class_name]
                    ))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, label

class DatasetManager:
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Enhanced training transforms with Albumentations
        self.train_transform = A.Compose([
            A.RandomResizedCrop(224, 224, scale=(0.7, 1.0)),
            A.OneOf([
                A.RandomRotate90(),
                A.Rotate(limit=45)
            ], p=0.5),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise()
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
            ], p=0.2),
            A.OneOf([
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Validation transform
        self.val_transform = A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Create datasets
        self.train_dataset = self._create_dataset(train=True)
        self.val_dataset = self._create_dataset(train=False)
    
    def _create_dataset(self, train: bool = True) -> FlowerDataset:
        transform = self.train_transform if train else self.val_transform
        return FlowerDataset(self.data_dir, transform=transform, train=train)
    
    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        # Use weighted sampler for training to handle class imbalance
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=self.train_dataset.sample_weights,
            num_samples=len(self.train_dataset),
            replacement=True
        )
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def get_num_classes(self) -> int:
        return len(self.train_dataset.classes)
    
    def get_class_names(self) -> list:
        return self.train_dataset.classes