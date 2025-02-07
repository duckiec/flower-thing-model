import torch
import torch.nn as nn
import torchvision.models as models
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
        
        # Use efficient-net-b0 for good accuracy/speed trade-off
        self.base_model = models.efficientnet_b0(pretrained=pretrained)
        
        # Freeze early layers
        for param in list(self.base_model.parameters())[:-20]:
            param.requires_grad = False
        
        # Simplified classifier head
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        for m in self.base_model.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

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