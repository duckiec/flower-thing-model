import os
import tarfile
import zipfile
import requests
from tqdm import tqdm
import shutil
import glob
from PIL import Image
import torchvision.datasets as datasets

def download_file(url, filename):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f, tqdm(
            desc=f"Downloading {os.path.basename(filename)}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        return True
    except Exception as e:
        print(f"\nError downloading {filename}: {str(e)}")
        if os.path.exists(filename):
            os.remove(filename)  # Remove partially downloaded file
        return False

def prepare_oxford_flowers():
    """Oxford 102 Flowers dataset"""
    dataset_path = 'datasets/oxford102'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path, exist_ok=True)
        success = download_file(
            'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz',
            f'{dataset_path}/102flowers.tgz'
        )
        if success:
            print("Extracting Oxford 102 Flowers dataset...")
            with tarfile.open(f'{dataset_path}/102flowers.tgz') as tar:
                tar.extractall(dataset_path)
            return True
        return False
    return True

def prepare_flowers_recognition():
    """Flowers Recognition dataset"""
    if not os.path.exists('datasets/flowers_recognition'):
        os.makedirs('datasets/flowers_recognition', exist_ok=True)
        # Note: This would normally download from Kaggle, but requires authentication
        # For this example, assume the user has downloaded the dataset manually
        print("Please download the Flowers Recognition dataset from Kaggle and place it in datasets/flowers_recognition")

def prepare_flower_photos():
    """TensorFlow Flowers dataset"""
    dataset_path = 'datasets/flower_photos'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path, exist_ok=True)
        success = download_file(
            'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
            f'{dataset_path}/flower_photos.tgz'
        )
        if success:
            print("Extracting TensorFlow Flowers dataset...")
            with tarfile.open(f'{dataset_path}/flower_photos.tgz') as tar:
                tar.extractall(dataset_path)
            return True
        return False
    return True

def prepare_flowers17():
    """Oxford 17 Flowers dataset"""
    dataset_path = 'datasets/oxford17'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path, exist_ok=True)
        success = download_file(
            'https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz',
            f'{dataset_path}/17flowers.tgz'
        )
        if success:
            print("Extracting Oxford 17 Flowers dataset...")
            with tarfile.open(f'{dataset_path}/17flowers.tgz') as tar:
                tar.extractall(dataset_path)
            return True
        return False
    return True

def prepare_flora_incognita():
    """Flora Incognita dataset (subset)"""
    if not os.path.exists('datasets/flora_incognita'):
        os.makedirs('datasets/flora_incognita', exist_ok=True)
        print("Please download the Flora Incognita dataset from their official source and place it in datasets/flora_incognita")

def organize_datasets():
    """Organize all datasets into a common structure"""
    target_dir = 'data/flowers'
    os.makedirs(target_dir, exist_ok=True)
    
    # Process each dataset and copy to the common structure
    datasets_to_process = [
        ('datasets/oxford102/jpg', 'oxford102'),
        ('datasets/flowers_recognition/*', 'recognition'),
        ('datasets/flower_photos/*', 'tensorflow'),
        ('datasets/oxford17/jpg', 'oxford17'),
        ('datasets/flora_incognita/*', 'flora')
    ]
    
    for src_pattern, dataset_prefix in datasets_to_process:
        for src_dir in glob.glob(src_pattern):
            if os.path.isdir(src_dir):
                class_name = os.path.basename(src_dir)
                dst_dir = os.path.join(target_dir, f"{dataset_prefix}_{class_name}")
                os.makedirs(dst_dir, exist_ok=True)
                
                # Copy and convert images
                for img_path in glob.glob(os.path.join(src_dir, '*.*')):
                    if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        try:
                            img = Image.open(img_path)
                            img = img.convert('RGB')
                            dst_path = os.path.join(dst_dir, os.path.basename(img_path))
                            img.save(dst_path, 'JPEG')
                        except Exception as e:
                            print(f"Error processing {img_path}: {e}")

def prepare_all_datasets():
    """Main function to prepare all datasets that can be imported"""
    print("\nPreparing all datasets...")
    
    # Create necessary directories
    os.makedirs('datasets', exist_ok=True)
    os.makedirs('data/flowers', exist_ok=True)
    
    # Track successfully downloaded datasets
    success_count = 0
    total_downloadable = 3  # Oxford 102, TensorFlow Flowers, Oxford 17
    
    # Download and prepare all datasets
    if prepare_oxford_flowers():
        success_count += 1
    if prepare_flower_photos():
        success_count += 1
    if prepare_flowers17():
        success_count += 1
    
    # Manual dataset notifications
    print("\nNOTE: Two additional datasets require manual download:")
    print("1. Flowers Recognition dataset from Kaggle:")
    print("   https://www.kaggle.com/datasets/alxmamaev/flowers-recognition")
    print("   Place it in: datasets/flowers_recognition/")
    print("2. Flora Incognita dataset:")
    print("   https://zenodo.org/records/3770768")
    print("   Place it in: datasets/flora_incognita/")
    
    # Organize datasets into common structure
    print(f"\nSuccessfully downloaded {success_count}/{total_downloadable} automatic datasets")
    print("Organizing available datasets...")
    organize_datasets()
    
    if success_count < total_downloadable:
        print("\nWARNING: Some datasets could not be downloaded automatically.")
        print("The model will still train with available datasets.")
    
    print("\nDataset preparation completed!")

if __name__ == "__main__":
    prepare_all_datasets()