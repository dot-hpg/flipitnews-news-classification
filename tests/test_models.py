import pytest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from flipitnews.models import make_models, make_vectorizer, train_and_evaluate

TRAIN = ["bank market share profit", "market share investor bank", "match goal coach win",
         "coach win cup match", "film actor award oscar", "award film actress festival"] * 3
LABELS = [0, 0, 1, 1, 2, 2] * 3


@pytest.mark.parametrize("choice, cls", [("bow", CountVectorizer), ("Bag of Words", CountVectorizer),
                                         ("tfidf", TfidfVectorizer), ("TF-IDF", TfidfVectorizer)])
def test_make_vectorizer_accepts_user_choice(choice, cls):
    assert type(make_vectorizer(choice)) is cls


def test_make_vectorizer_rejects_unknown_choice():
    with pytest.raises(ValueError):
        make_vectorizer("word2vec")


def test_required_models_are_present():
    assert {"Naive Bayes", "Decision Tree", "Nearest Neighbors", "Random Forest"} <= set(make_models())


@pytest.mark.parametrize("method", ["bow", "tfidf"])
def test_train_and_evaluate_learns_toy_topics(method):
    r = train_and_evaluate("Naive Bayes", make_models()["Naive Bayes"], TRAIN, LABELS,
                           ["profit bank", "goal coach", "oscar actor"], [0, 1, 2], method)
    assert r.accuracy == 1.0 and r.f1 == 1.0
    assert r.row()["vectorizer"] in ("Bag of Words", "TF-IDF")
