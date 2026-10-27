"""Load the trained model bundle and classify raw articles, with probabilities and an explanation."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from .preprocess import clean_text

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUNDLE = ROOT / "models" / "flipitnews_classifiers.joblib"


class NewsClassifier:
    """Wraps the bundle written by scripts/train.py: {pipelines, classes, default, metrics}."""

    def __init__(self, bundle: dict):
        self.pipelines = bundle["pipelines"]
        self.classes = list(bundle["classes"])
        self.default = bundle["default"]
        self.metrics = bundle.get("metrics", {})
        self.trained_on = bundle.get("trained_on", {})

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "NewsClassifier":
        return cls(joblib.load(path or os.environ.get("FLIPIT_MODEL_PATH", DEFAULT_BUNDLE)))

    @property
    def available(self) -> list[str]:
        return list(self.pipelines)

    def resolve(self, model: str | None = None, vectorizer: str | None = None) -> str:
        """Pick a pipeline key such as 'logistic_regression__tfidf'; either part defaults to the deployed model."""
        d_model, d_vec = self.default.split("__")
        key = f"{model or d_model}__{vectorizer or d_vec}"
        if key not in self.pipelines:
            raise KeyError(f"No model {key!r}. Available: {', '.join(self.available)}")
        return key

    def predict(self, texts: list[str], model: str | None = None, vectorizer: str | None = None,
                top_terms: int = 8) -> list[dict]:
        key = self.resolve(model, vectorizer)
        pipe = self.pipelines[key]
        cleaned = [clean_text(t) for t in texts]
        proba = pipe.predict_proba(cleaned)
        X = pipe.named_steps["vectorizer"].transform(cleaned)
        out = []
        for i, p in enumerate(proba):
            k = int(np.argmax(p))
            ranked = np.argsort(p)[::-1]
            out.append({
                "category": self.classes[k],
                "confidence": round(float(p[k]), 4),
                "probabilities": {self.classes[j]: round(float(p[j]), 4) for j in ranked},
                "top_terms": self._explain(pipe, X[i], k, top_terms),
                "model": key,
                "tokens_used": int(X[i].nnz),
            })
        return out

    @staticmethod
    def _explain(pipe, x_row, k: int, n: int) -> list[dict]:
        """Words that pushed the article towards the predicted class (linear models and Naive Bayes only)."""
        clf = pipe.named_steps["model"]
        if hasattr(clf, "coef_"):
            weights = clf.coef_[k]
        elif hasattr(clf, "feature_log_prob_"):
            lp = clf.feature_log_prob_
            weights = lp[k] - np.delete(lp, k, axis=0).mean(axis=0)
        else:
            return []
        idx, vals = x_row.indices, x_row.data * weights[x_row.indices]
        vocab = pipe.named_steps["vectorizer"].get_feature_names_out()
        order = np.argsort(vals)[::-1][:n]
        return [{"term": str(vocab[idx[j]]), "weight": round(float(vals[j]), 4)} for j in order if vals[j] > 0]


@lru_cache(maxsize=1)
def get_classifier() -> NewsClassifier:
    return NewsClassifier.load()
