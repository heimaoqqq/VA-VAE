"""
FID Evaluation Script for Per-User Generated Samples
Compares generated samples against real user dataset images
"""

import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
import argparse
from pathlib import Path
import json
from tqdm import tqdm


class ImageDataset(Dataset):
    """Simple dataset for loading images from a directory"""
    
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform or transforms.Compose([
            transforms.Resize((299, 299)),  # InceptionV3 input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy image in case of error
            return torch.zeros(3, 299, 299)


def collect_images_from_directory(directory, extensions=('.png', '.jpg', '.jpeg')):
    """Collect all image files from a directory"""
    image_paths = []
    for ext in extensions:
        image_paths.extend(Path(directory).glob(f'**/*{ext}'))
    return [str(p) for p in image_paths]


def collect_real_images_for_user(dataset_dir, user_id, extensions=('.png', '.jpg', '.jpeg')):
    """
    Collect real images for a specific user from the dataset
    Assumes dataset structure: dataset_dir/ID_X/ (where X = user_id + 1)
    """
    # 微多普勒数据集的ID映射：user_id -> ID_(user_id+1)
    # 例如：user_id=23 -> ID_24
    mapped_id = user_id + 1
    
    # Try different possible directory naming patterns
    possible_patterns = [
        f"ID_{mapped_id}",        # 主要模式：ID_24
        f"ID{mapped_id}",         # 备选：ID24
        f"user_{user_id}",        # 备选：user_23
        f"user{user_id}",         # 备选：user23
        f"{user_id}",             # 备选：23
        f"User_{user_id}",        # 备选：User_23
        f"User{user_id}"          # 备选：User23
    ]
    
    for pattern in possible_patterns:
        user_dir = Path(dataset_dir) / pattern
        if user_dir.exists():
            print(f"Found user directory: {user_dir}")
            return collect_images_from_directory(user_dir, extensions)
    
    # If no user-specific directory found, try to find images with user_id in filename
    all_images = collect_images_from_directory(dataset_dir, extensions)
    user_images = [img for img in all_images if f"user{user_id}" in Path(img).stem.lower() or f"user_{user_id}" in Path(img).stem.lower()]
    
    if user_images:
        print(f"Found {len(user_images)} images with user_id {user_id} in filename")
        return user_images
    else:
        print(f"Warning: No images found for user_id {user_id}")
        return []


def compute_fid_score(real_images_dir, generated_images_dir, batch_size=32, device='cuda'):
    """
    Compute FID score between real and generated images
    """
    print("Initializing FID computation...")
    
    # Collect image paths
    real_image_paths = collect_images_from_directory(real_images_dir)
    generated_image_paths = collect_images_from_directory(generated_images_dir)
    
    print(f"Real images: {len(real_image_paths)}")
    print(f"Generated images: {len(generated_image_paths)}")
    
    if len(real_image_paths) == 0 or len(generated_image_paths) == 0:
        raise ValueError("Need at least one image in both real and generated directories")
    
    # Initialize FID metric
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    
    # Create data loaders
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    
    real_dataset = ImageDataset(real_image_paths, transform)
    generated_dataset = ImageDataset(generated_image_paths, transform)
    
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    generated_loader = DataLoader(generated_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Process real images
    print("Processing real images...")
    for batch in tqdm(real_loader, desc="Real images"):
        # Convert to uint8 format expected by FID
        batch_uint8 = (batch * 255).clamp(0, 255).to(torch.uint8)
        fid.update(batch_uint8.to(device), real=True)
    
    # Process generated images
    print("Processing generated images...")
    for batch in tqdm(generated_loader, desc="Generated images"):
        # Convert to uint8 format expected by FID
        batch_uint8 = (batch * 255).clamp(0, 255).to(torch.uint8)
        fid.update(batch_uint8.to(device), real=False)
    
    # Compute FID score
    fid_score = fid.compute()
    return fid_score.item()


def evaluate_fid_for_user(user_id, real_dataset_dir, generated_samples_dir, 
                         output_file=None, batch_size=32, device='cuda'):
    """
    Evaluate FID score for a specific user
    """
    print(f"Evaluating FID for user_id: {user_id}")
    
    # Collect real images for this user
    real_images = collect_real_images_for_user(real_dataset_dir, user_id)
    
    if len(real_images) == 0:
        print(f"No real images found for user_id {user_id}")
        return None
    
    # Check if generated samples directory exists
    generated_dir = Path(generated_samples_dir)
    if not generated_dir.exists():
        print(f"Generated samples directory does not exist: {generated_dir}")
        return None
    
    # Create temporary directory with real user images
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as temp_real_dir:
        # Copy real images to temp directory
        for img_path in real_images:
            shutil.copy2(img_path, temp_real_dir)
        
        try:
            # Compute FID
            fid_score = compute_fid_score(temp_real_dir, generated_dir, batch_size, device)
            
            result = {
                'user_id': user_id,
                'fid_score': fid_score,
                'num_real_images': len(real_images),
                'num_generated_images': len(collect_images_from_directory(generated_dir)),
                'real_dataset_dir': str(real_dataset_dir),
                'generated_samples_dir': str(generated_dir)
            }
            
            print(f"FID Score for User {user_id}: {fid_score:.4f}")
            print(f"Real images: {result['num_real_images']}, Generated images: {result['num_generated_images']}")
            
            # Save result if output file specified
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Results saved to: {output_file}")
            
            return result
            
        except Exception as e:
            print(f"Error computing FID for user {user_id}: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description='Evaluate FID score for per-user generated samples')
    parser.add_argument('--user_id', type=int, required=True, help='User ID to evaluate')
    parser.add_argument('--real_dataset_dir', type=str, required=True, 
                       help='Directory containing real dataset images')
    parser.add_argument('--generated_samples_dir', type=str, required=True,
                       help='Directory containing generated samples for this user')
    parser.add_argument('--output_file', type=str, 
                       help='Output file to save FID results (JSON format)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Evaluate FID
    result = evaluate_fid_for_user(
        user_id=args.user_id,
        real_dataset_dir=args.real_dataset_dir,
        generated_samples_dir=args.generated_samples_dir,
        output_file=args.output_file,
        batch_size=args.batch_size,
        device=args.device
    )
    
    if result is None:
        print("FID evaluation failed")
        return 1
    
    return 0


if __name__ == "__main__":
    main()
