import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from typing import Tuple, Optional

class FlowerDataset(Dataset):
    def __init__(self, data_dir: str, transform: Optional[transforms.Compose] = None, train: bool = True):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        
        # Get all classes (flower types)
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._get_samples()
        
        print(f"Found {len(self.classes)} classes and {len(self.samples)} images")
    
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
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class DatasetManager:
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # GPU-optimized transforms for training
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.2, 0.2, 0.2)
            ], p=0.8),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Simple validation transforms
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        self.train_dataset = self._create_dataset(train=True)
        self.val_dataset = self._create_dataset(train=False)
    
    def _create_dataset(self, train: bool = True) -> FlowerDataset:
        transform = self.train_transform if train else self.val_transform
        return FlowerDataset(self.data_dir, transform=transform, train=train)
    
    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
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