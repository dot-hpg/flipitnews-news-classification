"""Train, evaluate and save every model the API and web app can serve.

1. Clean all articles with flipitnews.preprocess.clean_text (cached in data/processed/).
2. Stratified 75:25 split, train each model with Bag of Words and with TF-IDF, record held-out metrics.
3. Refit every pipeline on all unique articles and save one compressed bundle + a JSON model card.

Usage:  python scripts/train.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from flipitnews.models import SEED, make_models, make_pipeline, train_and_evaluate  # noqa: E402
from flipitnews.preprocess import clean_text  # noqa: E402

DATA = ROOT / "data" / "flipitnews-data.csv"
CACHE = ROOT / "data" / "processed" / "articles_clean.csv"
MODELS = ROOT / "models"
DEFAULT = "logistic_regression__tfidf"
SERVED = ["Naive Bayes", "Decision Tree", "Nearest Neighbors", "Random Forest", "Logistic Regression"]


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def load_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    if CACHE.exists():
        cached = pd.read_csv(CACHE)
        if len(cached) == len(df) and (cached.Article == df.Article).all():
            return cached
    print("Cleaning articles (about 40 s) ...")
    df["clean"] = df.Article.map(clean_text)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False)
    return df


def main() -> None:
    df = load_clean()
    enc = LabelEncoder()
    y = enc.fit_transform(df.Category)
    Xtr, Xte, ytr, yte = train_test_split(df.clean, y, test_size=0.25, stratify=y, random_state=SEED)

    metrics, pipelines = {}, {}
    unique = df.drop_duplicates(subset="Article")
    y_all = enc.transform(unique.Category)
    for method in ["bow", "tfidf"]:
        for name, model in make_models().items():
            if name not in SERVED:
                continue
            key = f"{slug(name)}__{method}"
            r = train_and_evaluate(name, model, Xtr, ytr, Xte, yte, method)
            metrics[key] = {"model": name, "vectorizer": method, "accuracy": round(r.accuracy, 4),
                            "precision_macro": round(r.precision, 4), "recall_macro": round(r.recall, 4),
                            "f1_macro": round(r.f1, 4)}
            print(f"{key:34s} accuracy {r.accuracy:.4f}  macro F1 {r.f1:.4f}")
            # Final model: same settings, refit on every unique article so nothing is wasted.
            pipelines[key] = make_pipeline(make_models()[name], method).fit(unique.clean, y_all)

    bundle = {"pipelines": pipelines, "classes": list(enc.classes_), "default": DEFAULT, "metrics": metrics,
              "trained_on": {"articles": int(len(unique)), "date": dt.date.today().isoformat(),
                             "sklearn": sklearn.__version__}}
    MODELS.mkdir(exist_ok=True)
    path = MODELS / "flipitnews_classifiers.joblib"
    joblib.dump(bundle, path, compress=3)
    card = {k: v for k, v in bundle.items() if k != "pipelines"}
    card["evaluation"] = "Held-out stratified 25% test split (557 articles), random_state=42"
    (MODELS / "model_card.json").write_text(json.dumps(card, indent=2))
    print(f"Saved {len(pipelines)} pipelines to {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
