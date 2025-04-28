#!/usr/bin/env python3
"""
Ensemble AI
Date: Apr 24, 2025

This script trains a standard CNN model on the CIFAR-10 dataset using the
same hyperparameters as the NdLinear version, and saves the training loss
and test accuracy curves to a PDF.

Usage Example:

python src/cnn_img_baseline.py \
  --batch_size 64 \
  --learning_rate 0.001 \
  --epochs 20 \
  --data_dir './data' \
  --output_file 'baseline_results.pdf'
"""

import argparse
import logging
import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


def get_args():
    parser = argparse.ArgumentParser(description="Baseline CNN Training Script")
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Batch size for training and testing (default: 64)'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=0.001,
        help='Learning rate for the optimizer (default: 0.001)'
    )
    parser.add_argument(
        '--epochs', type=int, default=20,
        help='Number of training epochs (default: 20)'
    )
    parser.add_argument(
        '--data_dir', type=str, default='./data',
        help='Directory for the CIFAR-10 dataset (default: ./data)'
    )
    parser.add_argument(
        '--output_file', type=str, default='baseline_results.pdf',
        help='Output PDF file for saving training curves (default: baseline_results.pdf)'
    )
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()]
    )


def get_device():
    """Select GPU if available, else fall back to CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        logging.info("No GPU found, using CPU.")
        return torch.device('cpu')


def get_transform():
    """Compose the standard CIFAR-10 normalization pipeline."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])


def load_data(data_dir, batch_size):
    """Download CIFAR-10 and wrap in DataLoader for train and test splits."""
    transform = get_transform()
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return trainloader, testloader


class BaselineVision(nn.Module):
    """
    A simple CNN for CIFAR-10 classification to mirror the NdLinear architecture,
    but using standard nn.Linear layers.
    """
    def __init__(self):
        super(BaselineVision, self).__init__()
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.relu  = nn.ReLU(inplace=True)
        # After two pools, spatial dims go from 32→16→8
        flattened_dim = 64 * 8 * 8
        # Fully connected classifier
        # Match the hidden dimension used in NdLinear example: 32*8*8 = 2048
        self.fc1    = nn.Linear(flattened_dim, flattened_dim)
        self.fc_out = nn.Linear(flattened_dim, 10)

    def forward(self, x):
        # Extract features
        x = self.pool(self.relu(self.conv1(x)))  # [B,32,16,16]
        x = self.pool(self.relu(self.conv2(x)))  # [B,64, 8, 8]
        # Flatten
        x = x.view(x.size(0), -1)                # [B,64*8*8]
        # Classifier
        x = self.relu(self.fc1(x))               # [B,64*8*8]
        x = self.fc_out(x)                       # [B,10]
        return x


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train the model for one epoch and return average loss."""
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


def evaluate(model, loader, device):
    """Evaluate the model on the test set and return accuracy (%)."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def plot_and_save(losses, accs, params, epochs, filename):
    """Plot training loss and test accuracy curves, then save to PDF."""
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), losses, label=f"BaselineVision ({params} params)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), accs, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Test Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    # Parse command-line arguments
    args = get_args()
    setup_logging()

    # Device and data
    device = get_device()
    trainloader, testloader = load_data(args.data_dir, args.batch_size)

    # Model, loss, optimizer
    model = BaselineVision().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Parameter count
    params = count_parameters(model)
    logging.info(f"Number of parameters: {params}")

    # Training loop
    losses, accs = [], []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, trainloader, criterion, optimizer, device)
        acc  = evaluate(model, testloader, device)
        losses.append(loss)
        accs.append(acc)
        logging.info(f"Epoch {epoch}/{args.epochs}  Loss: {loss:.4f}  Acc: {acc:.2f}%")

    # Plot & save curves
    plot_and_save(losses, accs, params, args.epochs, args.output_file)
    logging.info(f"Saved training curves to {args.standard_cnn_output_file}")


if __name__ == '__main__':
    main()
