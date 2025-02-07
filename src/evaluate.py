import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
from tqdm import tqdm
import pandas as pd

from model import create_model, get_hardware_info
from dataset import DatasetManager

def evaluate_model(model_path, data_dir):
    """
    Comprehensive model evaluation with detailed metrics and visualizations
    """
    # Get hardware settings
    hw_info = get_hardware_info()
    device = hw_info['device']
    batch_size = hw_info['optimal_batch_size']
    num_workers = hw_info['optimal_workers']
    
    print(f"\nEvaluating on: {device}")
    
    # Load data
    dataset_manager = DatasetManager(data_dir, batch_size=batch_size, num_workers=num_workers)
    _, val_loader = dataset_manager.get_loaders()
    class_names = dataset_manager.get_class_names()
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = create_model(len(class_names), device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Evaluation
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\nRunning evaluation...")
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Save classification report
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv('results/classification_report.csv')
    
    # Plot and save confusion matrix
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    plt.close()
    
    # Plot class-wise accuracy
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    plt.figure(figsize=(15, 8))
    sns.barplot(x=list(range(len(class_names))), y=class_accuracies)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/class_accuracies.png')
    plt.close()
    
    # Print summary
    print("\nEvaluation Results:")
    print(f"Overall Accuracy: {report['accuracy']:.4f}")
    print(f"Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1-Score: {report['weighted avg']['f1-score']:.4f}")
    
    return {
        'confusion_matrix': cm,
        'classification_report': report,
        'class_accuracies': class_accuracies
    }

if __name__ == "__main__":
    model_path = "model/best_model.pth"
    data_dir = "data/flowers"
    evaluate_model(model_path, data_dir)