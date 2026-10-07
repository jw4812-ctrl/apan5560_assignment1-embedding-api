FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /uvx /bin/

WORKDIR /code

COPY pyproject.toml uv.lock /code/
RUN uv sync --frozen --no-dev

COPY ./app /code/app
COPY ./models /code/models

EXPOSE 80

CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]