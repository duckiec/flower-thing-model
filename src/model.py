import torch
import torch.nn as nn
import torchvision.models as models
import timm
import os

def get_hardware_info():
    """Detect available hardware and optimize settings"""
    if torch.cuda.is_available():
        device = "cuda:0"
        # Check for tensor cores (available in Volta, Turing, Ampere, Ada architectures)
        gpu_cap = torch.cuda.get_device_capability()
        has_tensor_cores = gpu_cap[0] >= 7  # Volta and newer architectures
        
        if has_tensor_cores:
            torch.backends.cudnn.benchmark = True
            optimal_batch_size = 64  # Larger batch size for tensor cores
        else:
            optimal_batch_size = 32
        
        optimal_workers = min(8, torch.cuda.device_count() * 4)
    else:
        device = "cpu"
        optimal_batch_size = 16
        optimal_workers = min(4, os.cpu_count() or 1)
    
    return {
        'device': device,
        'optimal_batch_size': optimal_batch_size,
        'optimal_workers': optimal_workers,
        'use_amp': torch.cuda.is_available()  # Use AMP if CUDA is available
    }

class FlowerClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(FlowerClassifier, self).__init__()
        
        # Use EfficientNetV2 Small for better accuracy/efficiency ratio
        self.base_model = timm.create_model('efficientnetv2_s', pretrained=pretrained)
        
        # Progressive layer unfreezing
        self.frozen_layers = len(list(self.base_model.parameters())) - 30
        self.current_epoch = 0
        self.unfreeze_epoch = 5  # Start unfreezing after 5 epochs
        
        # Freeze initial layers
        for i, param in enumerate(self.base_model.parameters()):
            if i < self.frozen_layers:
                param.requires_grad = False
        
        # Enhanced classifier head with regularization
        in_features = self.base_model.get_classifier().in_features
        self.base_model.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )
        
        # Weight initialization
        for m in self.base_model.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def unfreeze_layers(self, epoch):
        """Progressive layer unfreezing"""
        self.current_epoch = epoch
        if epoch >= self.unfreeze_epoch:
            layers_to_unfreeze = min(20, (epoch - self.unfreeze_epoch) * 2)
            start_idx = self.frozen_layers - layers_to_unfreeze
            
            for i, param in enumerate(self.base_model.parameters()):
                if i >= start_idx:
                    param.requires_grad = True

    def forward(self, x):
        return self.base_model(x)

def create_model(num_classes, device=None):
    if device is None:
        device = get_hardware_info()['device']
    
    model = FlowerClassifier(num_classes=num_classes)
    model = model.to(device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    return model