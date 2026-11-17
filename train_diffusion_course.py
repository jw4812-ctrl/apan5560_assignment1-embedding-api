from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.diffusion_course import DiffusionModel, UNet


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}", flush=True)

    # Scale CIFAR-10 images from [0, 1] to [-1, 1].
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        ),
    ])

    dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )

    model = DiffusionModel(
        network=UNet(image_size=32, num_channels=3),
        ema_decay=0.8,
    ).to(device)
    
    model.load_state_dict(
        torch.load(
            "models/diffusion_course_epoch1.pth",
            map_location=device,
            weights_only=True,
        )
    )
    print("Loaded the epoch 1 model.", flush=True)

    optimizer = torch.optim.Adam(
        model.network.parameters(),
        lr=0.0002,
    )

    epochs = 5
    Path("models").mkdir(exist_ok=True)

    for epoch in range(epochs):
        model.network.train()
        model.ema_network.eval()
        total_loss = 0.0

        for batch_index, (images, _) in enumerate(loader, start=1):
            images = images.to(device)

            loss = model.training_loss(images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.update_ema()

            total_loss += loss.item()

            if batch_index % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Batch {batch_index}/{len(loader)} | "
                    f"Loss: {loss.item():.4f}",
                    flush=True,
                )

        print(
            f"Epoch {epoch + 1}/{epochs} completed | "
            f"Average loss: {total_loss / len(loader):.4f}",
            flush=True,
        )

        # Save both the training network and EMA network.
        torch.save(
            model.state_dict(),
            "models/diffusion_course_cifar10.pth",
        )

        print(
            "Saved models/diffusion_course_cifar10.pth",
            flush=True,
        )


if __name__ == "__main__":
    main()