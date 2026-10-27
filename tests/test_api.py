"""End-to-end checks against the trained model bundle (run scripts/train.py first)."""
import pytest
from fastapi.testclient import TestClient

from api.main import app
from flipitnews.inference import DEFAULT_BUNDLE

pytestmark = pytest.mark.skipif(not DEFAULT_BUNDLE.exists(), reason="model bundle not trained")

ARTICLES = {
    "Business": "Shares fell as the central bank raised interest rates and investors worried about company profits.",
    "Sports": "The striker scored twice as the champions won the cup final and the coach praised his team.",
    "Technology": "The new smartphone software lets mobile users browse the internet faster on broadband networks.",
    "Politics": "The prime minister faced questions in parliament as the opposition party attacked the election plan.",
    "Entertainment": "The actress won an oscar for the film, and the director thanked the cast at the awards.",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_models_lists_metrics(client):
    body = client.get("/models").json()
    assert body["default"] == "logistic_regression__tfidf"
    assert len(body["classes"]) == 5
    assert body["metrics"]["logistic_regression__tfidf"]["accuracy"] > 0.95


def test_predicts_every_category(client):
    r = client.post("/predict", json={"texts": list(ARTICLES.values())})
    assert r.status_code == 200
    assert [p["category"] for p in r.json()] == list(ARTICLES)
    first = r.json()[0]
    assert abs(sum(first["probabilities"].values()) - 1) < 1e-3
    assert first["top_terms"]


@pytest.mark.parametrize("model", ["naive_bayes", "decision_tree", "nearest_neighbors", "random_forest"])
@pytest.mark.parametrize("vectorizer", ["bow", "tfidf"])
def test_every_served_model_and_vectorizer(client, model, vectorizer):
    r = client.post("/predict", json={"texts": [ARTICLES["Sports"]], "model": model, "vectorizer": vectorizer})
    assert r.status_code == 200
    assert r.json()[0]["model"] == f"{model}__{vectorizer}"


def test_validation_errors(client):
    assert client.post("/predict", json={"texts": []}).status_code == 422
    assert client.post("/predict", json={"texts": ["   "]}).status_code == 422
    assert client.post("/predict", json={"texts": ["x"], "vectorizer": "word2vec"}).status_code == 422
