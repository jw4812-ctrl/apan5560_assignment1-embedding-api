# Assignment 2: CNN Image Classification API

This project adds a CIFAR-10 image classifier to the FastAPI word embedding API from Assignment 1.

## Run with Docker

Install and start Docker Desktop. From the project directory, run:

```bash
docker build -t assignment2-cnn-api .
docker run --rm -p 8000:80 assignment2-cnn-api
```

Open http://127.0.0.1:8000/docs to test the endpoints.

The trained weights are included in `models/cnn_cifar10.pth`, so retraining is not required to run the API. Inference runs on the CPU.

## Image classification

Endpoint: `POST /classify`

Upload an image using the multipart form field `file`.

In Swagger UI:
1. Expand `POST /classify`.
2. Click **Try it out**.
3. Select an image file.
4. Click **Execute**.

Alternatively, replace `image.jpg` with the path to your image:

```bash
curl -X POST http://127.0.0.1:8000/classify \
  -F "file=@image.jpg"
```

The response contains:
- `class_id`: predicted class index, from 0 to 9.
- `class`: predicted class name.
- `confidence`: softmax probability for the predicted class.

Classes, in index order: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

Images are converted to RGB, resized to 64 × 64, and converted to tensors using the same preprocessing as training.

## CNN architecture

| Layer | Output shape, excluding batch |
|---|---|
| RGB input | 3 × 64 × 64 |
| Conv2d: 16 filters, kernel 3, stride 1, padding 1; ReLU | 16 × 64 × 64 |
| MaxPool2d: kernel 2, stride 2 | 16 × 32 × 32 |
| Conv2d: 32 filters, kernel 3, stride 1, padding 1; ReLU | 32 × 32 × 32 |
| MaxPool2d: kernel 2, stride 2 | 32 × 16 × 16 |
| Flatten | 8192 |
| Linear; ReLU | 100 |
| Linear | 10 |

## Training

The model uses the CIFAR-10 training set with:
- Cross-entropy loss
- Adam optimizer with learning rate 0.0005
- Batch size 32
- 10 epochs
- PyTorch random seed 42

The trained model achieved **66.00% accuracy** on the CIFAR-10 test set.

To retrain with Python 3.11 and uv installed:

```bash
uv sync --python 3.11 --frozen
uv run python train_cnn.py
```

The script downloads CIFAR-10 into `data/` and saves weights to `models/cnn_cifar10.pth`. It uses MPS or CUDA when available, otherwise CPU.

## Run locally

```bash
uv sync --python 3.11 --frozen
uv run uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs.

## Existing embedding endpoint

Endpoint: `POST /embedding`

```bash
curl -X POST http://127.0.0.1:8000/embedding \
  -H "Content-Type: application/json" \
  -d '{"word":"hello"}'
```

Returns the word and its spaCy embedding.

## Project files

- `app/cnn_model.py`: CNN architecture
- `train_cnn.py`: model training and evaluation
- `app/main.py`: FastAPI endpoints and image inference
- `app/embedding_model.py`: word embedding functionality
- `models/cnn_cifar10.pth`: trained CNN weights
- `Dockerfile`: Docker deployment
- `pyproject.toml` and `uv.lock`: dependencies