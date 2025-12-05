import os
import torch
import torch.nn as nn
from data_loader import get_data_loaders
from model import build_model



def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy


if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ROOT = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(ROOT, "data", "processed")

    # Load dataset
    _, _, test_loader, class_names = get_data_loaders(
        data_dir=data_dir,
        batch_size=32,
        num_workers=4
    )

    # Build model
    num_classes = len(class_names)
    model = build_model(num_classes)
    model = model.to(device)

    # Load checkpoint
    checkpoint_path = os.path.join(ROOT, "best_model.pth")
    print(f"Loading checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Evaluate
    print("\nEvaluating on test set...")
    test_loss, test_acc = evaluate(model, test_loader, device)

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")