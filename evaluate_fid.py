"""
FID Evaluation Script - Unified Version
Uses PyTorch and scipy to compute FID score directly
No external dependencies like torch-fidelity needed
"""

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import inception_v3
import argparse
from pathlib import Path
import json
from tqdm import tqdm
from scipy.linalg import sqrtm


class InceptionV3FeatureExtractor(nn.Module):
    """Extract features from InceptionV3 for FID calculation"""
    
    def __init__(self, device='cuda'):
        super().__init__()
        # Load pre-trained InceptionV3
        self.inception = inception_v3(pretrained=True, transform_input=False)
        # Remove the final classifier to get 2048-dim features
        self.inception.fc = nn.Identity()
        self.inception.eval()
        self.inception.to(device)
        self.device = device
        
    def forward(self, x):
        # x should be [0,1] range tensor of shape (B,3,299,299)
        # InceptionV3 expects input in range [-1, 1]
        x = 2 * x - 1  # Convert [0,1] to [-1,1]
        
        with torch.no_grad():
            # Get pool3 features (2048-dim)
            features = self.inception(x)
            return features


class ImageDataset(Dataset):
    """Simple dataset for loading images"""
    
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform or transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
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
    mapped_id = user_id + 1
    
    # Try different possible directory naming patterns
    possible_patterns = [
        f"ID_{mapped_id}",
        f"ID{mapped_id}",
        f"user_{user_id}",
        f"user{user_id}",
        f"{user_id}",
        f"User_{user_id}",
        f"User{user_id}"
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


def extract_features_from_images(image_paths, feature_extractor, batch_size=32, device='cuda'):
    """Extract InceptionV3 features from a list of image paths"""
    
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    
    dataset = ImageDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    features = []
    
    for batch in tqdm(dataloader, desc="Extracting features"):
        batch = batch.to(device)
        with torch.no_grad():
            batch_features = feature_extractor(batch)
            features.append(batch_features.cpu().numpy())
    
    return np.concatenate(features, axis=0)


def calculate_fid_score(real_features, generated_features):
    """
    Calculate FID score from feature vectors
    FID = ||mu1 - mu2||^2 + Tr(C1 + C2 - 2*sqrt(C1*C2))
    """
    
    # Calculate statistics
    mu1 = np.mean(real_features, axis=0)
    mu2 = np.mean(generated_features, axis=0)
    
    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(generated_features, rowvar=False)
    
    # Add small epsilon for numerical stability
    eps = 1e-6
    sigma1 = sigma1 + eps * np.eye(sigma1.shape[0])
    sigma2 = sigma2 + eps * np.eye(sigma2.shape[0])
    
    # Calculate FID
    diff = mu1 - mu2
    
    # Product of covariance matrices
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real
    
    # Numerical stability
    tr_covmean = np.trace(covmean)
    
    fid = (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
    
    return fid


def compute_fid_score_standalone(real_images_dir, generated_images_dir, batch_size=32, device='cuda'):
    """
    Compute FID score using standalone implementation
    """
    print("Initializing standalone FID computation...")
    
    # Collect image paths
    real_image_paths = collect_images_from_directory(real_images_dir)
    generated_image_paths = collect_images_from_directory(generated_images_dir)
    
    print(f"Real images: {len(real_image_paths)}")
    print(f"Generated images: {len(generated_image_paths)}")
    
    if len(real_image_paths) == 0 or len(generated_image_paths) == 0:
        raise ValueError("Need at least one image in both real and generated directories")
    
    # Initialize feature extractor
    print("Loading InceptionV3 feature extractor...")
    feature_extractor = InceptionV3FeatureExtractor(device)
    
    # Extract features
    print("Extracting features from real images...")
    real_features = extract_features_from_images(real_image_paths, feature_extractor, batch_size, device)
    
    print("Extracting features from generated images...")  
    generated_features = extract_features_from_images(generated_image_paths, feature_extractor, batch_size, device)
    
    # Calculate FID score
    print("Computing FID score...")
    fid_score = calculate_fid_score(real_features, generated_features)
    
    return fid_score


def evaluate_fid_for_user_standalone(user_id, real_dataset_dir, generated_samples_dir, 
                                   output_file=None, batch_size=32, device='cuda'):
    """
    Evaluate FID score for a specific user using standalone implementation
    """
    print(f"Evaluating FID for user_id: {user_id} (Standalone)")
    
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
            fid_score = compute_fid_score_standalone(temp_real_dir, generated_dir, batch_size, device)
            
            result = {
                'user_id': user_id,
                'fid_score': float(fid_score),
                'num_real_images': len(real_images),
                'num_generated_images': len(collect_images_from_directory(generated_dir)),
                'real_dataset_dir': str(real_dataset_dir),
                'generated_samples_dir': str(generated_dir),
                'method': 'standalone'
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
    parser = argparse.ArgumentParser(description='Evaluate FID score using standalone implementation')
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
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Evaluate FID
    result = evaluate_fid_for_user_standalone(
        user_id=args.user_id,
        real_dataset_dir=args.real_dataset_dir,
        generated_samples_dir=args.generated_samples_dir,
        output_file=args.output_file,
        batch_size=args.batch_size,
        device=device
    )
    
    if result is None:
        print("FID evaluation failed")
        return 1
    
    return 0


if __name__ == "__main__":
    main()
