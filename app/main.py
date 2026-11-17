from io import BytesIO
from pathlib import Path
from app.gan_model import Generator, LATENT_DIM

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from torchvision import transforms

from app.cnn_model import CNN
from app.embedding_model import calculate_embedding

from app.energy_model import EnergyModel, langevin_sample
from app.diffusion_course import DiffusionModel, UNet

app = FastAPI()

# Load the trained model on the CPU for deployment.
model_path = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "cnn_cifar10.pth"
)

cnn = CNN()
cnn.load_state_dict(
    torch.load(model_path, map_location="cpu", weights_only=True)
)
cnn.eval()

gan_model_path = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "gan_mnist.pth"
)

generator = Generator()
generator.load_state_dict(
    torch.load(gan_model_path, map_location="cpu", weights_only=True)
)
generator.eval()

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Use the same preprocessing as training.
image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


class EmbeddingRequest(BaseModel):
    word: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    embedding = calculate_embedding(request.word)
    return {
        "word": request.word,
        "embedding": embedding.tolist(),
    }


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    contents = await file.read()

    try:
        with Image.open(BytesIO(contents)) as image:
            image = image.convert("RGB")
            inputs = image_transform(image).unsqueeze(0)
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image.",
        )

    with torch.no_grad():
        outputs = cnn(inputs)
        probabilities = torch.softmax(outputs, dim=1)
        class_id = probabilities.argmax(dim=1).item()

    return {
        "class_id": class_id,
        "class": classes[class_id],
        "confidence": probabilities[0, class_id].item(),
    }

@app.get("/generate")
def generate_images(num_samples: int = 4):
    if num_samples < 1 or num_samples > 16:
        raise HTTPException(
            status_code=400,
            detail="num_samples must be between 1 and 16.",
        )

    noise = torch.randn(num_samples, LATENT_DIM)

    with torch.no_grad():
        images = generator(noise)
        images = (images.squeeze(1) + 1) / 2

    return {
        "num_samples": num_samples,
        "image_shape": [28, 28],
        "images": images.tolist(),
    }

# Load the CIFAR-10 energy model on the CPU.
energy_model_path = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "energy_cifar10_buffer.pth"
)

energy_model = EnergyModel()
energy_model.load_state_dict(
    torch.load(energy_model_path, map_location="cpu", weights_only=True)
)
energy_model.eval()

# Load the CIFAR-10 diffusion model on the CPU.
diffusion_model_path = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "diffusion_course_cifar10.pth"
)

diffusion_model = DiffusionModel(
    network=UNet(image_size=32, num_channels=3),
    ema_decay=0.8,
)
diffusion_model.load_state_dict(
    torch.load(diffusion_model_path, map_location="cpu", weights_only=True)
)
diffusion_model.eval()


@app.get("/generate-energy")
def generate_energy_images(num_samples: int = 1):
    if num_samples < 1 or num_samples > 4:
        raise HTTPException(
            status_code=400,
            detail="num_samples must be between 1 and 4.",
        )

    initial_images = torch.rand(num_samples, 3, 32, 32) * 2 - 1

    # Langevin sampling needs gradients with respect to the images.
    images = langevin_sample(
        energy_model,
        initial_images,
        steps=400,
        step_size=1.0,
        noise_scale=0.005,
    )
    images = ((images + 1) / 2).clamp(0, 1)

    return {
        "num_samples": num_samples,
        "image_shape": [3, 32, 32],
        "images": images.tolist(),
    }


@app.get("/generate-diffusion")
def generate_diffusion_images(num_samples: int = 1):
    if num_samples < 1 or num_samples > 4:
        raise HTTPException(
            status_code=400,
            detail="num_samples must be between 1 and 4.",
        )

    images = diffusion_model.generate(
        num_images=num_samples,
        diffusion_steps=100,
        image_size=32,
    )

    return {
        "num_samples": num_samples,
        "image_shape": [3, 32, 32],
        "images": images.tolist(),
    }