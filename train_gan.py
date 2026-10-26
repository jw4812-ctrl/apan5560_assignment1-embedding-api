from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from app.gan_model import Discriminator, Generator, LATENT_DIM


def main():
    torch.manual_seed(42)

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}", flush=True)

    EPOCHS = 10
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0002

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    d_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    Path("models").mkdir(exist_ok=True)

    for epoch in range(EPOCHS):
        generator.train()
        discriminator.train()

        total_d_loss = 0.0
        total_g_loss = 0.0

        for real_images, _ in train_loader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # Train the discriminator on real images and detached fake images.
            d_optimizer.zero_grad()

            real_predictions = discriminator(real_images)
            real_labels = torch.ones_like(real_predictions)
            real_loss = criterion(real_predictions, real_labels)

            noise = torch.randn(batch_size, LATENT_DIM, device=device)
            fake_images = generator(noise).detach()
            fake_predictions = discriminator(fake_images)
            fake_labels = torch.zeros_like(fake_predictions)
            fake_loss = criterion(fake_predictions, fake_labels)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            d_optimizer.step()

            # Train the generator to make fake images look real to the discriminator.
            g_optimizer.zero_grad()

            noise = torch.randn(batch_size, LATENT_DIM, device=device)
            generated_images = generator(noise)
            predictions = discriminator(generated_images)
            generator_labels = torch.ones_like(predictions)

            g_loss = criterion(predictions, generator_labels)
            g_loss.backward()
            g_optimizer.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()

        print(
            f"Epoch {epoch + 1}/{EPOCHS}, "
            f"D loss={total_d_loss / len(train_loader):.4f}, "
            f"G loss={total_g_loss / len(train_loader):.4f}",
            flush=True,
        )

    torch.save(generator.state_dict(), "models/gan_mnist.pth")
    print("Generator saved to models/gan_mnist.pth", flush=True)


if __name__ == "__main__":
    main()