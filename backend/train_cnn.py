"""
Train Character Recognition CNN on EMNIST Balanced
====================================================
Optimized for CPU training with strong data augmentation.

Usage:
    cd backend && python train_cnn.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import time
from sklearn.metrics import confusion_matrix

from model import CharacterCNN, EMNIST_LABELS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def fix_emnist_orientation(img):
    """EMNIST images are transposed. Fix for batch [B, C, H, W] or single [C, H, W]."""
    if img.dim() == 3:
        return img.transpose(1, 2).flip(2)
    return img.transpose(2, 3).flip(3)


def load_data(batch_size=256):
    """Download and load EMNIST Balanced with augmentation."""
    print('Loading EMNIST Balanced dataset...')

    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.08, 0.08),
            scale=(0.9, 1.1),
            shear=8,
        ),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.EMNIST(
        root='../data', split='balanced', train=True, download=True,
        transform=train_transform
    )
    test_dataset = datasets.EMNIST(
        root='../data', split='balanced', train=False, download=True,
        transform=test_transform
    )

    print(f'  Training: {len(train_dataset):,} | Test: {len(test_dataset):,} | Classes: {len(EMNIST_LABELS)}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    return train_loader, test_loader


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images = fix_emnist_orientation(images)
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / total, 100.0 * correct / total


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = fix_emnist_orientation(images)
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return 100.0 * correct / total, np.array(all_preds), np.array(all_labels)


def save_plots(train_losses, train_accs, test_accs, all_preds, all_labels):
    os.makedirs('../outputs', exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, 'b-', linewidth=2, label='Train')
    ax2.plot(epochs, test_accs, 'r-', linewidth=2, label='Test')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../outputs/training_curves.png', dpi=150)
    plt.close()
    print('Saved: outputs/training_curves.png')

    cm = confusion_matrix(all_labels, all_preds)
    group_labels = ['0-9', 'A-Z', 'a-z']
    group_ranges = [(0, 10), (10, 36), (36, 47)]
    group_cm = np.zeros((3, 3), dtype=int)
    for i, (si, ei) in enumerate(group_ranges):
        for j, (sj, ej) in enumerate(group_ranges):
            group_cm[i, j] = cm[si:ei, sj:ej].sum()

    plt.figure(figsize=(8, 6))
    sns.heatmap(group_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=group_labels, yticklabels=group_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Grouped)')
    plt.tight_layout()
    plt.savefig('../outputs/confusion_matrix.png', dpi=150)
    plt.close()
    print('Saved: outputs/confusion_matrix.png')


def main():
    print('=' * 60)
    print('  CharacterAI CNN v2 - EMNIST Balanced')
    print('  Optimized Architecture + Data Augmentation')
    print('=' * 60)
    print(f'  Device: {device}\n')

    os.makedirs('../models', exist_ok=True)

    train_loader, test_loader = load_data(batch_size=256)

    model = CharacterCNN(num_classes=47).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'  Model parameters: {total_params:,}')

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-5)

    num_epochs = 15
    best_acc = 0
    train_losses, train_accs, test_accs = [], [], []

    print(f'  Training for {num_epochs} epochs...\n')
    for epoch in range(1, num_epochs + 1):
        start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        test_acc, _, _ = evaluate(model, test_loader)
        scheduler.step()

        elapsed = time.time() - start
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        lr = optimizer.param_groups[0]['lr']
        print(f'  Epoch {epoch:2d}/{num_epochs} | '
              f'Loss: {train_loss:.4f} | '
              f'Train: {train_acc:.2f}% | '
              f'Test: {test_acc:.2f}% | '
              f'LR: {lr:.6f} | '
              f'Time: {elapsed:.1f}s')

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_classes': 47,
                'accuracy': best_acc,
                'epoch': epoch,
            }, '../models/character_cnn.pth')
            print(f'         -> Best model saved ({best_acc:.2f}%)')

    # Final eval
    print(f'\n  Loading best model ({best_acc:.2f}%)...')
    checkpoint = torch.load('../models/character_cnn.pth', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    final_acc, all_preds, all_labels = evaluate(model, test_loader)
    print(f'  Final Test Accuracy: {final_acc:.2f}%')

    save_plots(train_losses, train_accs, test_accs, all_preds, all_labels)

    for name, s, e in [('Digits', 0, 10), ('Uppercase', 10, 36), ('Lowercase', 36, 47)]:
        mask = (all_labels >= s) & (all_labels < e)
        if mask.sum() > 0:
            acc = (all_preds[mask] == all_labels[mask]).mean() * 100
            print(f'  {name} Accuracy: {acc:.2f}%')

    print(f'\n  DONE! Best Accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()
