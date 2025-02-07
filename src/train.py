import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
from datetime import datetime
import os

from dataset import DatasetManager
from model import create_model, get_hardware_info
from check_hardware import check_hardware
from prepare_datasets import prepare_all_datasets
from evaluate import evaluate_model

def setup_environment():
    """Prepare environment and datasets"""
    print("\n=== Checking Hardware Configuration ===")
    check_hardware()
    
    print("\n=== Preparing Datasets ===")
    prepare_all_datasets()

def train_model(data_dir, num_epochs=30):
    # Get hardware settings
    hw_info = get_hardware_info()
    device = hw_info['device']
    batch_size = hw_info['optimal_batch_size']
    num_workers = hw_info['optimal_workers']
    use_amp = hw_info['use_amp']
    
    print(f"\n=== Starting Training ===")
    print(f"Training on: {device}")
    print(f"Using mixed precision: {use_amp}")
    
    # Initialize dataset and model
    dataset_manager = DatasetManager(data_dir, batch_size=batch_size, num_workers=num_workers)
    train_loader, val_loader = dataset_manager.get_loaders()
    model = create_model(dataset_manager.get_num_classes(), device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=3e-3, epochs=num_epochs, steps_per_epoch=len(train_loader))
    scaler = GradScaler() if use_amp else None
    
    # Training history
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = train_correct = train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass with automatic mixed precision
            if use_amp:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            # Calculate metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = val_correct = val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # Calculate epoch metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('model', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'model/best_model.pth')
    
    plot_training_history(history)
    return history, model

def plot_training_history(history):
    plt.style.use('seaborn')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Val Accuracy')
    ax2.set_title('Accuracy Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'plots/training_history_{timestamp}.png')
    plt.close()

def main():
    """Main entry point that handles everything"""
    try:
        # Step 1: Setup environment and prepare datasets
        setup_environment()
        
        # Step 2: Train the model
        data_dir = "data/flowers"
        history, model = train_model(data_dir)
        
        # Step 3: Evaluate the final model
        print("\n=== Evaluating Model ===")
        evaluate_model("model/best_model.pth", data_dir)
        
        print("\n=== Training Pipeline Completed Successfully ===")
        print("Check the 'results' directory for evaluation metrics and visualizations")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()