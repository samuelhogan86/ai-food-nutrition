#train the model from model.py - cannot work here until model.py is done
import os
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_data_loaders
from model import build_model
#small values for batch and epochs just to make it easier on my laptop.
batch_size = 32      #originally ran at 8, would like to have 32
num_epochs = 15      #originally ran at 1, would like to have 15
learning_rate = 0.001

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


ROOT = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(ROOT, "data", "processed")

train_loader, val_loader, test_loader, class_names = get_data_loaders(
    data_dir=data_dir,
    batch_size=batch_size,
    num_workers=4  #set to 0 for testing on cpu
)

num_classes = len(class_names)

model = build_model(num_classes=num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train_one_epoch(model, dataloader, criterion, optimizer, device):

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        print("MODEL:", next(model.parameters()).device)
        print("INPUT:", images.device)
        print("LABELS:", labels.device)
        break

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        #print progress
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
            print(f"Batch [{batch_idx+1}/{len(dataloader)}], "
                  f"Loss: {running_loss/10:.4f}, "
                  f"Accuracy: {100.*correct/total:.2f}%")
            running_loss = 0.0

    epoch_loss = criterion(outputs, labels).item()
    epoch_acc = 100.*correct/total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss /= len(dataloader)
    val_acc = 100.*correct/total
    return val_loss, val_acc

def main():

    print("----- STARTING TRAINING -----\n")

    checkpoint_path = os.path.join(ROOT, "best_model.pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading existing checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("No previous checkpoint found. Starting fresh.")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"\nEpoch {epoch+1} summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%\n")

    save_path = os.path.join(ROOT, "best_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")

if __name__ == "__main__":
    main()