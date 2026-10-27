"""Vectorizers, classifiers and the train-and-evaluate routine shared by the notebook and scripts."""
from __future__ import annotations

import io
import pickle
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
VECTORIZERS = {"bow": "Bag of Words", "tfidf": "TF-IDF"}

# Shared vocabulary settings so BoW and TF-IDF are compared on exactly the same features:
# words that appear in at least 2 and at most 95% of training articles, unigrams + bigrams, top 5,000.
VOCAB = dict(min_df=2, max_df=0.95, ngram_range=(1, 2), max_features=5000)


def make_vectorizer(method: str = "tfidf"):
    """The user's choice of text representation: 'bow' (raw counts) or 'tfidf' (weighted, L2-normalised)."""
    method = method.lower().replace("-", "").replace(" ", "")
    if method in ("bow", "bagofwords", "count"):
        return CountVectorizer(**VOCAB)
    if method == "tfidf":
        return TfidfVectorizer(sublinear_tf=True, **VOCAB)
    raise ValueError(f"Unknown vectorizer {method!r}; choose 'bow' or 'tfidf'.")


def make_models() -> dict:
    """The four required classifiers plus two linear baselines that are standard for text."""
    return {
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "Nearest Neighbors": KNeighborsClassifier(n_neighbors=7, metric="cosine", weights="distance"),
        "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
        "Logistic Regression": LogisticRegression(C=10, max_iter=2000),
        "Linear SVM": LinearSVC(C=0.5, random_state=SEED),
    }


def make_pipeline(model, method: str = "tfidf") -> Pipeline:
    return Pipeline([("vectorizer", make_vectorizer(method)), ("model", model)])


@dataclass
class Result:
    name: str
    vectorizer: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    train_seconds: float
    predict_ms_per_doc: float
    size_kb: float
    y_pred: np.ndarray = field(repr=False)
    pipeline: Pipeline = field(repr=False)

    def row(self) -> dict:
        return {"model": self.name, "vectorizer": VECTORIZERS[self.vectorizer], "accuracy": self.accuracy,
                "precision": self.precision, "recall": self.recall, "f1_macro": self.f1,
                "train_s": self.train_seconds, "predict_ms/doc": self.predict_ms_per_doc, "size_kb": self.size_kb}


def train_and_evaluate(name, model, X_train, y_train, X_test, y_test, method="tfidf") -> Result:
    """Fit vectorizer + model on the training articles only, then score on the held-out test articles."""
    pipe = make_pipeline(model, method)
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_pred = pipe.predict(X_test)
    pred_ms = 1000 * (time.perf_counter() - t0) / len(X_test)
    buf = io.BytesIO()
    pickle.dump(pipe, buf)
    return Result(name, method, accuracy_score(y_test, y_pred),
                  precision_score(y_test, y_pred, average="macro"), recall_score(y_test, y_pred, average="macro"),
                  f1_score(y_test, y_pred, average="macro"), train_s, pred_ms, buf.tell() / 1024, y_pred, pipe)


def report(result: Result, y_test, class_names) -> tuple[str, np.ndarray]:
    return (classification_report(y_test, result.y_pred, target_names=class_names, digits=3),
            confusion_matrix(y_test, result.y_pred))
