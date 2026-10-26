# Assignment 3: MNIST GAN API

This project extends the Module 6 FastAPI application with a PyTorch GAN trained on MNIST. It keeps the existing word embedding and CIFAR-10 classification endpoints and adds an endpoint for generating handwritten digit images.

## Run with Docker

From the project directory, build and start the API:

```bash
docker build -t apan5560-assignment3 .
docker run --rm -p 8000:80 apan5560-assignment3
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to try the endpoints. The Docker image includes both trained model checkpoints, so retraining is not required to run inference.

## Generate MNIST images

Endpoint: `GET /generate`

Optional query parameter: `num_samples`, an integer from 1 to 16 (default: 4).

Example: `http://127.0.0.1:8000/generate?num_samples=4`

The response contains the number of samples, image dimensions (`28 × 28`), and generated pixel values scaled to `[0, 1]`.

## GAN architecture

- **Generator:** accepts a 100-dimensional noise vector, projects it to `128 × 7 × 7`, then uses two transposed convolution layers to produce a `1 × 28 × 28` image. Batch normalization and ReLU follow the first layer; Tanh is used at the output.
- **Discriminator:** takes a `1 × 28 × 28` image through two convolution layers, with BatchNorm and LeakyReLU as specified, then outputs a single real/fake probability.

## Train the GAN

With Python 3.11 and `uv` installed, run:

```bash
uv sync --python 3.11 --frozen
uv run python train_gan.py
```

The script downloads MNIST into `data/`, trains for 10 epochs, and saves the generator weights to `models/gan_mnist.pth`. It uses MPS or CUDA when available, otherwise CPU. The discriminator is used during training; only the generator is needed for image generation in the API.

## Existing API endpoints

- `POST /embedding`: returns a spaCy word embedding.
- `POST /classify`: accepts an image upload and returns a CIFAR-10 class prediction.
- `GET /generate`: generates MNIST-like digit images.

## Project files

- `app/gan_model.py`: GAN Generator and Discriminator architectures.
- `train_gan.py`: MNIST GAN training script.
- `models/gan_mnist.pth`: trained Generator weights used by `/generate`.
- `app/main.py`: FastAPI application and endpoints.
- `app/cnn_model.py`, `train_cnn.py`, and `models/cnn_cifar10.pth`: existing CIFAR-10 classifier.
- `app/embedding_model.py`: existing word embedding functionality.
- `Dockerfile`: Docker deployment configuration.
- `pyproject.toml` and `uv.lock`: Python dependencies.
