from torch import nn


LATENT_DIM = 100


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(LATENT_DIM, 7 * 7 * 128)

        self.deconv1 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(
            64, 1, kernel_size=4, stride=2, padding=1
        )
        self.tanh = nn.Tanh()

    def forward(self, noise):
        x = self.fc(noise)
        x = x.view(x.size(0), 128, 7, 7)
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.tanh(self.deconv2(x))
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=4, stride=2, padding=1
        )
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(
            64, 128, kernel_size=4, stride=2, padding=1
        )
        self.bn2 = nn.BatchNorm2d(128)
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image):
        x = self.leaky_relu1(self.conv1(image))
        x = self.leaky_relu2(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        x = self.sigmoid(self.fc(x))
        return x