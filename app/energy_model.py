import torch
from torch import nn


class EnergyModel(nn.Module):
    """Assign an energy score to each 32 × 32 RGB image."""

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),

            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1),
        )

    def forward(self, images):
        return self.network(images).squeeze(-1)


def langevin_sample(model, images, steps=40, step_size=0.01, noise_scale=0.005):
    """Sample lower-energy images using Langevin dynamics."""
    samples = images.detach()

    for _ in range(steps):
        # Add noise without tracking its operations.
        with torch.no_grad():
            samples = (
                samples + noise_scale * torch.randn_like(samples)
            ).clamp(-1.0, 1.0)

        # Compute gradients with respect to the images.
        samples = samples.detach().requires_grad_(True)
        with torch.enable_grad():
            energy = model(samples).sum()
            gradient = torch.autograd.grad(energy, samples)[0]

        # Clip image gradients, then update the images.
        with torch.no_grad():
            gradient = gradient.clamp(-0.03, 0.03)
            samples = (samples - step_size * gradient).clamp(-1.0, 1.0)

    return samples.detach()