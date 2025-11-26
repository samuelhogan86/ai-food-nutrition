import argparse
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Simple logger for script output
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class BasicBlock(nn.Module):
    """
    Basic residual block used by ResNet-style models.

    Contains two 3x3 convolutions with BatchNorm and a residual shortcut.
    If the spatial size or channel count changes (via `stride` or `in_channels` !=
    `out_channels`), the shortcut performs a 1x1 convolution projection.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    """
    A compact ResNet-18-like model composed of `BasicBlock` modules.

    Note: this implementation uses a 3x3 stem (common for smaller images)
    instead of the original ResNet 7x7 stem.
    """
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def train(model, trainloader, criterion, optimizer, device, epochs=10):
    """
    Train loop for the provided model and dataloader.

    This is intentionally simple (no scheduler, no validation). Checkpointing
    and logging are handled by the caller (see `__main__`).
    """
    model.to(device)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(trainloader) if len(trainloader) > 0 else float('nan')
        logger.info(f'Epoch {epoch} loss: {avg_loss}')
        # Yield epoch number and average loss to caller for checkpointing/logging
        yield epoch, avg_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet18 on Food dataset')
    parser.add_argument('--data-dir', default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed', 'train'),
                        help='Path to training data (ImageFolder style)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--checkpoint-dir', default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints'))
    parser.add_argument('--resume', default=None, help='Path to checkpoint to resume from')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Resolve and validate dataset directory
    train_dir = args.data_dir
    if not os.path.isdir(train_dir):
        raise RuntimeError(f"Train directory not found: {train_dir}. Please create it or pass a DataLoader.")

    # Data transforms (standard ImageNet normalization)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(root=train_dir, transform=transform)

    # Device selection: prefer CUDA, then MPS (Apple Metal), then CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    logger.info(f'Using device: {device}')
    if device.type == 'cuda':
        try:
            logger.info('CUDA device name: %s', torch.cuda.get_device_name(0))
        except Exception:
            pass

    # Create checkpoint directory if needed
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # pin_memory is beneficial for CUDA transfers; not required for MPS/CPU
    pin_memory = (device.type == 'cuda')
    trainloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)

    model = ResNet18(num_classes=len(dataset.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    start_epoch = 0
    # Optionally resume from checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info('Loading checkpoint %s', args.resume)
            ckpt = torch.load(args.resume, map_location='cpu')
            model.load_state_dict(ckpt.get('model_state_dict', ckpt))
            optimizer.load_state_dict(ckpt.get('optimizer_state_dict', optimizer.state_dict()))
            start_epoch = ckpt.get('epoch', 0) + 1
            logger.info('Resumed from epoch %d', start_epoch)
        else:
            logger.warning('Resume checkpoint not found: %s', args.resume)

    # Main train loop with checkpointing
    for epoch, avg_loss in train(model, trainloader, criterion, optimizer, device, epochs=args.epochs):
        # Save checkpoint after each epoch
        ckpt_path = os.path.join(args.checkpoint_dir, f'resnet18_epoch{epoch}.pth')
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss}, ckpt_path)
        logger.info('Saved checkpoint: %s', ckpt_path)