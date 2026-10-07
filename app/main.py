from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from torchvision import transforms

from app.cnn_model import CNN
from app.embedding_model import calculate_embedding

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

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Use the same preprocessing as training.
image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5),
    ),
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