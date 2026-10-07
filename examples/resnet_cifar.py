"""ResNet on CIFAR-10, comparing AnyInit against the stock initialization.

Downloads CIFAR-10 into ./data on first run and uses CUDA when available.

Run: python examples/resnet_cifar.py --epochs 3
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

import anyinit


def loaders(batch_size: int, workers: int = 2):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=train_transform
    )
    test = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True, transform=test_transform
    )
    return (
        torch.utils.data.DataLoader(train, batch_size, shuffle=True, num_workers=workers),
        torch.utils.data.DataLoader(test, batch_size, shuffle=False, num_workers=workers),
    )


def build(num_classes: int = 10) -> nn.Module:
    """ResNet-18 adapted to 32x32 inputs: the stock stem throws away too much detail."""
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def accuracy(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            predicted = model(images.to(device)).argmax(1)
            correct += int((predicted == labels.to(device)).sum())
            total += labels.size(0)
    return 100.0 * correct / total


def run(
    use_anyinit: bool, epochs: int, batch_size: int, device: torch.device, seed: int = 0
) -> float:
    torch.manual_seed(seed)
    model = build()

    if use_anyinit:
        report = anyinit.initialize(model, input_spec=(batch_size, 3, 32, 32), seed=seed)
        print(report)
        print()

    model = model.to(device)
    train_loader, test_loader = loaders(batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.1, total_steps=epochs * len(train_loader)
    )

    for epoch in range(epochs):
        model.train()
        started, running = time.time(), 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            schedule.step()
            running += float(loss)
        print(
            f"  epoch {epoch + 1}/{epochs}  loss {running / len(train_loader):.4f}  "
            f"test {accuracy(model, test_loader, device):.2f}%  ({time.time() - started:.0f}s)"
        )
    return accuracy(model, test_loader, device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    print("=== stock PyTorch initialization ===")
    baseline = run(False, args.epochs, args.batch_size, device)
    print("\n=== AnyInit ===")
    tuned = run(True, args.epochs, args.batch_size, device)

    print(f"\nfinal test accuracy: stock {baseline:.2f}%  AnyInit {tuned:.2f}%")


if __name__ == "__main__":
    main()
