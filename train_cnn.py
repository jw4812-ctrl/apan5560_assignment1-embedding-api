from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.cnn_model import CNN


def main():
    torch.manual_seed(42)

    # Use the Mac GPU if available; otherwise use the CPU.
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}", flush=True)

    # The assignment requires 64 x 64 RGB inputs.
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        ),
    ])

    train_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )
    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_data, batch_size=64, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_data, batch_size=64, shuffle=False, num_workers=0
    )

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    Path("models").mkdir(exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_number, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

            if batch_number % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"batch {batch_number}/{len(train_loader)}",
                    flush=True,
                )

        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"loss={total_loss / total:.4f}, "
            f"train accuracy={100 * correct / total:.2f}%",
            flush=True,
        )

        # Save after every epoch, with weights stored on the CPU.
        weights = {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        }
        torch.save(weights, "models/cnn_cifar10.pth")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    print(f"Test accuracy: {100 * correct / total:.2f}%", flush=True)
    print("Model saved to models/cnn_cifar10.pth", flush=True)


if __name__ == "__main__":
    main()