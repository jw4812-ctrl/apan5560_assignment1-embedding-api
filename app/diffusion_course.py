import copy
import math

import torch
from torch import nn
import torch.nn.functional as F


def cosine_schedule(diffusion_times):
    """Calculate noise and signal rates for continuous diffusion times."""
    min_signal_rate = 0.02
    max_signal_rate = 0.95

    start_angle = math.acos(max_signal_rate)
    end_angle = math.acos(min_signal_rate)

    angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = torch.cos(angles)
    noise_rates = torch.sin(angles)

    return noise_rates, signal_rates


class SinusoidalEmbedding(nn.Module):
    def __init__(self, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies

        frequencies = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(1000.0),
                num_frequencies,
            )
        )

        self.register_buffer(
            "angular_speeds",
            2.0 * math.pi * frequencies.view(1, 1, 1, -1),
        )

    def forward(self, x):
        x = x.expand(-1, 1, 1, self.num_frequencies)

        sin_part = torch.sin(self.angular_speeds * x)
        cos_part = torch.cos(self.angular_speeds * x)

        return torch.cat([sin_part, cos_part], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        if in_channels != out_channels:
            self.proj = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            )
        else:
            self.proj = nn.Identity()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        residual = self.proj(x)

        x = F.silu(self.conv1(x))
        x = self.conv2(x)

        return x + residual


class DownBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super().__init__()

        self.blocks = nn.ModuleList()

        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels, width))
            in_channels = width

        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)

        return self.pool(x)


class UpBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super().__init__()

        self.blocks = nn.ModuleList()

        for _ in range(block_depth):
            self.blocks.append(
                ResidualBlock(in_channels + width, width)
            )
            in_channels = width

    def forward(self, x, skips):
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        for block in self.blocks:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        return x


class UNet(nn.Module):
    def __init__(self, image_size=32, num_channels=3, embedding_dim=32):
        super().__init__()

        self.initial = nn.Conv2d(
            num_channels,
            32,
            kernel_size=1,
        )

        self.num_channels = num_channels
        self.image_size = image_size
        self.embedding_dim = embedding_dim

        self.embedding = SinusoidalEmbedding(num_frequencies=16)

        self.down1 = DownBlock(
            width=32,
            in_channels=64,
            block_depth=2,
        )

        self.down2 = DownBlock(
            width=64,
            in_channels=32,
            block_depth=2,
        )

        self.down3 = DownBlock(
            width=96,
            in_channels=64,
            block_depth=2,
        )

        self.mid1 = ResidualBlock(
            in_channels=96,
            out_channels=128,
        )

        self.mid2 = ResidualBlock(
            in_channels=128,
            out_channels=128,
        )

        self.up1 = UpBlock(
            width=96,
            in_channels=128,
            block_depth=2,
        )

        self.up2 = UpBlock(
            width=64,
            in_channels=96,
            block_depth=2,
        )

        self.up3 = UpBlock(
            width=32,
            in_channels=64,
            block_depth=2,
        )

        self.final = nn.Conv2d(
            32,
            num_channels,
            kernel_size=1,
        )

        nn.init.zeros_(self.final.weight)

    def forward(self, noisy_images, noise_variances):
        skips = []

        x = self.initial(noisy_images)

        # Convert noise variance into a spatial embedding.
        noise_embedding = self.embedding(noise_variances)
        noise_embedding = noise_embedding.permute(0, 3, 1, 2)

        noise_embedding = F.interpolate(
            noise_embedding,
            size=(self.image_size, self.image_size),
            mode="nearest",
        )

        x = torch.cat([x, noise_embedding], dim=1)

        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)

        x = self.mid1(x)
        x = self.mid2(x)

        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)

        return self.final(x)

class DiffusionModel(nn.Module):
    def __init__(self, network, ema_decay=0.8):
        super().__init__()

        self.network = network
        self.ema_network = copy.deepcopy(network)
        self.ema_network.eval()

        # The instructor's example uses 0.8.
        self.ema_decay = ema_decay

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        if training:
            network = self.network
        else:
            network = self.ema_network

        # The U-Net receives the noise variance.
        predicted_noise = network(noisy_images, noise_rates ** 2)

        # Estimate the clean image.
        predicted_image = (
            noisy_images - noise_rates * predicted_noise
        ) / signal_rates

        return predicted_noise, predicted_image

    def training_loss(self, images):
        batch_size = images.size(0)

        # Images are expected to be normalized to [-1, 1].
        noise = torch.randn_like(images)

        diffusion_times = torch.rand(
            batch_size, 1, 1, 1, device=images.device
        )

        noise_rates, signal_rates = cosine_schedule(diffusion_times)

        noisy_images = (
            signal_rates * images
            + noise_rates * noise
        )

        predicted_noise, _ = self.denoise(
            noisy_images,
            noise_rates,
            signal_rates,
            training=True,
        )

        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def update_ema(self):
        for ema_param, param in zip(
            self.ema_network.parameters(),
            self.network.parameters(),
        ):
            ema_param.mul_(self.ema_decay)
            ema_param.add_(param, alpha=1.0 - self.ema_decay)

    @torch.no_grad()
    def generate(self, num_images=1, diffusion_steps=100, image_size=32):
        device = next(self.network.parameters()).device
        self.ema_network.eval()

        current_images = torch.randn(
            num_images,
            self.network.num_channels,
            image_size,
            image_size,
            device=device,
        )

        step_size = 1.0 / diffusion_steps

        for step in range(diffusion_steps):
            t = torch.full(
                (num_images, 1, 1, 1),
                1.0 - step * step_size,
                device=device,
            )

            noise_rates, signal_rates = cosine_schedule(t)

            predicted_noise, predicted_image = self.denoise(
                current_images,
                noise_rates,
                signal_rates,
                training=False,
            )

            next_t = t - step_size
            next_noise_rates, next_signal_rates = cosine_schedule(next_t)

            current_images = (
                next_signal_rates * predicted_image
                + next_noise_rates * predicted_noise
            )

        # Convert from [-1, 1] to [0, 1] for saving or display.
        return ((predicted_image + 1.0) / 2.0).clamp(0.0, 1.0)