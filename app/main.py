from fastapi import FastAPI
from pydantic import BaseModel
from app.embedding_model import calculate_embedding

app = FastAPI()


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
        "embedding": embedding.tolist()
    }