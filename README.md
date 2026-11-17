# Assignment 4: Advanced Image Generation API

This project extends the existing FastAPI application with two image
generators trained on CIFAR-10: an energy-based model (EBM) and a diffusion
model. It also retains the word embedding, CIFAR-10 classification, and
MNIST GAN endpoints from previous assignments.

## Run with Docker

From the project directory, build and start the API:

```bash
docker build -t assignment4-api .
docker run --rm -p 8001:80 assignment4-api
```

Open http://localhost:8001/docs to try the endpoints.

The repository includes the trained model weights needed for inference;
retraining is not required to run the API.

## Assignment 4 endpoints

### Energy-based image generation

Endpoint: `GET /generate-energy`

Optional parameter: `num_samples`, an integer from 1 to 4 (default: 1).

The endpoint starts from random images and uses 400 steps of Langevin
sampling to generate low-energy images.

Example:

```text
http://localhost:8001/generate-energy?num_samples=1
```

### Diffusion image generation

Endpoint: `GET /generate-diffusion`

Optional parameter: `num_samples`, an integer from 1 to 4 (default: 1).

The endpoint starts from random noise and uses the trained EMA network
with 100 reverse diffusion steps to generate images.

Example:

```text
http://localhost:8001/generate-diffusion?num_samples=1
```

Both endpoints return JSON containing `num_samples`, `image_shape`, and
image pixel values. The image shape is `[3, 32, 32]` in channel-first
order, and pixel values are scaled to `[0, 1]`.

Generation runs on the CPU in Docker, so a request may take some time.

## Models and training

Both new models were trained on the CIFAR-10 training dataset.

- **Energy model:** A convolutional network assigns an energy score to
  an image. Training uses contrastive divergence, a sample buffer, and
  Langevin sampling. The deployed weights are
  `models/energy_cifar10_buffer.pth`.
- **Diffusion model:** A U-Net predicts noise at different diffusion
  times. The implementation uses a cosine noise schedule and an
  exponential moving average (EMA) network for generation. The deployed
  weights are `models/diffusion_course_cifar10.pth`.

Generated samples are low-resolution and can be blurry.

### Run training

With Python 3.11 or later and `uv` installed:

```bash
uv sync --frozen
uv run python train_energy.py
uv run python train_diffusion_course.py
```

The scripts download CIFAR-10 into `data/` if needed and save weights
to the `models/` directory. Running a training script overwrites its
corresponding deployed weights.

The diffusion model included in the repository was trained for one epoch,
then for four additional epochs after reloading its weights. The optimizer
was reinitialized for the second run. The script is configured to train
from scratch for five epochs, so it may not produce identical results.

## Existing endpoints

- `POST /embedding`: returns a spaCy word embedding.
- `POST /classify`: accepts an image upload and predicts a CIFAR-10 class.
- `GET /generate`: generates MNIST-like digit images.

## Project files

- `app/energy_model.py`: energy network and Langevin sampling.
- `train_energy.py`: CIFAR-10 energy model training script.
- `app/diffusion_course.py`: U-Net, diffusion process, and EMA model.
- `train_diffusion_course.py`: CIFAR-10 diffusion training script.
- `app/main.py`: FastAPI application and endpoints.
- `models/`: trained model weights used by the API.
- `Dockerfile`: Docker deployment configuration.
- `pyproject.toml` and `uv.lock`: Python dependencies.

