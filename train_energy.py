from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.energy_model import EnergyModel, langevin_sample


class SampleBuffer:
    """Keep generated images so sampling can continue in later batches."""

    def __init__(self, device, size=8192):
        self.device = device
        self.size = size

        # Initial buffer: random CIFAR-10-sized RGB images.
        self.samples = torch.rand(128, 3, 32, 32) * 2 - 1

    def sample(self, model, batch_size, steps, step_size, noise_scale):
        # Draw existing samples from the buffer.
        indices = torch.randint(0, len(self.samples), (batch_size,))
        initial_images = self.samples[indices].clone()

        # Replace approximately 5% with new random images.
        new_mask = torch.rand(batch_size) < 0.05
        count_new = int(new_mask.sum().item())

        if count_new > 0:
            initial_images[new_mask] = (
                torch.rand(count_new, 3, 32, 32) * 2 - 1
            )

        initial_images = initial_images.to(self.device)

        generated = langevin_sample(
            model,
            initial_images,
            steps=steps,
            step_size=step_size,
            noise_scale=noise_scale,
        )

        # Keep the buffer on CPU and limit its maximum size.
        self.samples = torch.cat(
            [generated.detach().cpu(), self.samples],
            dim=0,
        )[:self.size]

        return generated


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}", flush=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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

    model = EnergyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    buffer = SampleBuffer(device=device)

    epochs = 10
    sampling_steps = 40

    # These are starting values; they are not confirmed instructor settings.
    step_size = 1.0
    noise_scale = 0.005
    alpha = 0.001

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_index, (real_images, _) in enumerate(loader, start=1):
            real_images = real_images.to(device)

            # Add a small amount of noise to real images.
            real_images = (
                real_images + noise_scale * torch.randn_like(real_images)
            ).clamp(-1.0, 1.0)

            fake_images = buffer.sample(
                model,
                batch_size=real_images.size(0),
                steps=sampling_steps,
                step_size=step_size,
                noise_scale=noise_scale,
            )

            real_energy = model(real_images)
            fake_energy = model(fake_images)

            contrastive_loss = real_energy.mean() - fake_energy.mean()
            regularization = alpha * (
                real_energy.square().mean() + fake_energy.square().mean()
            )
            loss = contrastive_loss + regularization

            optimizer.zero_grad()
            loss.backward()

            # Limit the size of parameter gradients.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

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

        # Use a different filename to preserve the previous model.
        Path("models").mkdir(exist_ok=True)
        torch.save(model.state_dict(), "models/energy_cifar10_buffer.pth")
        print("Saved models/energy_cifar10_buffer.pth", flush=True)


if __name__ == "__main__":
    main()