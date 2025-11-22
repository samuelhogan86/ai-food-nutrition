import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def get_train_transforms():
    """
    Training transformations with DATA AUGMENTATION

    Used: 
    - RandomResizedCrop: Zoom in/out on an image randomly
    - RandomHorizontalFlip: Mirror the image 50% of the time
    - ColorJitter: Vary image brightness, contrast, saturation
    - Normalize: Normalize using ImageNet Statistics
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=.2,
            contrast=.2,
            saturation=.2,
            hue=.1
        ),

        transforms.ToTensor(),

        #Normalize using ImageNet statistics since the mdoel is pretrained on ImageNet
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    ])

def get_test_transforms():
    """
    Validation/Test transformations

    Used:
    - Resize to 256x256
    - Center crop to 224x224
    - Convert to tensor
    - Normalize using ImageNet Statistics
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_data_loaders(data_dir, batch_size=32, num_workers=4):
    """
    Create DataLoaders for train, validation, and test sets

    Parameters:
    data_dir: Path to processed data directory
    batch_size: Number of images to load at once
    num_workers: Number of subprocesses for data loading 
                - speeds data loading by loading in parallel
    
    Returns:
    train_loader, val_loader, test_loader, class_names
    """

    data_dir = os.path.abspath(data_dir)

    print(f"Loading data from: {data_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Number of workers: {num_workers}")

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir,'val' )
    test_dir = os.path.join(data_dir, 'test')

    for split_name, split_dir in [('train', train_dir), ('val', val_dir), ('test', test_dir)]:
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"{split_name} directory not found at {split_dir}")
        
    print("\nLoading training data")
    train_dataset = datasets.ImageFolder(
        root = train_dir,
        transform = get_train_transforms()
    )

    print("\nLoading validation data")
    val_dataset = datasets.ImageFolder(
        root = val_dir,
        transform = get_test_transforms()
    )

    print("\nLoading test data")
    test_dataset = datasets.ImageFolder(
        root = test_dir,
        transform = get_train_transforms()
    )

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print(f"\nDataset loaded successfully!")
    print(f"\nNumber of classes: {num_classes}")
    print(f"\nTraining images: {len(train_dataset):,}")
    print(f"\nValidation images: {len(val_dataset):,}")
    print(f"\nTest images: {len(test_dataset):,}")
    print(f"\nSample classes: {class_names[:5]}...")

    #Creating DataLoaders - provide batching, shuffling, parallel loading
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, #Shuffle training data each epoch
        num_workers=num_workers,
        pin_memory=True # Speeds up GPU transfer
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, # Don't shuffle validation
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, class_names

def test_data_loader():
    #Verifies data loads correctly
    print("\nTesting data loader")

    ROOT = os.path.dirname(os.path.dirname(__file__))
    data_dir=os.path.join(ROOT, "data", "processed")

    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        data_dir=data_dir,
        batch_size=16,
        num_workers=0 #set to 0 for testing simplicity
    )

    print("Testing batch loading")

    images, labels = next(iter(train_loader))

    print(f"\nBatch shape: {images.shape}")
    print(f"  - Batch size: {images.shape[0]}")
    print(f"  - Channels: {images.shape[1]} (RGB)")
    print(f"  - Height: {images.shape[2]} pixels")
    print(f"  - Width: {images.shape[3]} pixels")
    print(f"\nLabels shape: {labels.shape}")
    print(f"  - Contains class indices (0-{len(class_names)-1})")

    print(f"\nImage tensor value range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  (Should be roughly [-2, 2] after normalization)")

    print(f"\nFirst 8 images in this batch:")
    for i in range(min(8, len(labels))):
        class_ind = labels[i].item()
        class_name = class_names[class_ind]
        print(f"  Image {i+1}: {class_name} (class index: {class_ind})")

    print("VISUALIZING SAMPLE IMAGES")

    def denormalize(tensor):
        #Reverse the normalization applied previously for display
        mean = torch.tensor([.485, .456, .406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        return tensor
    
    fig, axes = plt.subplots(2, 4, figsize=(12,6))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            # Convert from (C, H, W) to (H, W, C) for matplotlib
            img = denormalize(images[i]).permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.set_title(class_names[labels[i].item()])
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('data_loader_test.png', dpi=150, bbox_inches='tight')
    print("\nSaved visualization to: data_loader_test.png")
    plt.show()
    #close window for program to end
    
    print("\nData loader test complete")

if __name__ == "__main__":
    test_data_loader()