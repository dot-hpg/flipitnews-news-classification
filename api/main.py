"""FlipItNews category API.

    uvicorn api.main:app --port 8000        # docs at http://localhost:8000/docs
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from flipitnews import __version__
from flipitnews.inference import get_classifier
from flipitnews.preprocess import ensure_nltk_data

@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_nltk_data()
    get_classifier()          # load the model once, before the first request
    yield


app = FastAPI(title="FlipItNews Category API", version=__version__, lifespan=lifespan,
              description="Classifies news articles into Business, Entertainment, Politics, Sports or Technology.")

ModelName = Literal["naive_bayes", "decision_tree", "nearest_neighbors", "random_forest", "logistic_regression"]


class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=64, description="1 to 64 raw article texts")
    model: ModelName | None = Field(None, description="Defaults to the deployed model (logistic_regression)")
    vectorizer: Literal["bow", "tfidf"] | None = Field(None, description="Bag of Words or TF-IDF; default tfidf")

    model_config = {"json_schema_extra": {"examples": [{
        "texts": ["The central bank raised interest rates as inflation climbed and shares fell."],
        "vectorizer": "tfidf"}]}}


class Term(BaseModel):
    term: str
    weight: float


class Prediction(BaseModel):
    category: str
    confidence: float
    probabilities: dict[str, float]
    top_terms: list[Term]
    model: str
    tokens_used: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/models")
def models() -> dict:
    clf = get_classifier()
    return {"default": clf.default, "classes": clf.classes, "trained_on": clf.trained_on, "metrics": clf.metrics}


@app.post("/predict", response_model=list[Prediction])
def predict(req: PredictRequest) -> list[dict]:
    if any(len(t) > 100_000 for t in req.texts):
        raise HTTPException(413, "Each article must be under 100,000 characters.")
    if any(not t.strip() for t in req.texts):
        raise HTTPException(422, "Articles must not be empty.")
    try:
        return get_classifier().predict(req.texts, req.model, req.vectorizer)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
