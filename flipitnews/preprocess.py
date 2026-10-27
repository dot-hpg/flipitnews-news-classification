"""Text processing for news articles: non-letters -> tokens -> stopwords -> lemmas.

The same function is used in the notebook, the training script, the API and the web app,
so the text a model sees at inference time is processed exactly as it was during training.
"""
from __future__ import annotations

import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

NLTK_RESOURCES = {
    "corpora/stopwords": "stopwords",
    "tokenizers/punkt_tab": "punkt_tab",
    "corpora/wordnet": "wordnet",
    "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
}

NON_LETTERS = re.compile(r"[^a-z]+")
MIN_TOKEN_LEN = 2  # single letters left over from "u.s." or "it's" carry no topic signal
# WordNet strips a final "s" from these as if they were plurals: boss -> bos, us -> u, bonus stays bonus
# only by luck. Words ending in -ss / -us are left as they are.
KEEP_AS_IS = re.compile(r"(ss|us)$")


def ensure_nltk_data() -> None:
    """Download the NLTK resources this module needs, once, if they are missing."""
    for path, name in NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


# NLTK lists "won" as the stem of "won't", but in news it is mostly the past tense of "win" (a strong Sports word).
KEEP_WORDS = {"won"}


@lru_cache(maxsize=1)
def stop_words() -> frozenset[str]:
    ensure_nltk_data()
    return frozenset(stopwords.words("english")) - KEEP_WORDS


@lru_cache(maxsize=1)
def _lemmatizer() -> WordNetLemmatizer:
    ensure_nltk_data()
    return WordNetLemmatizer()


def _wordnet_pos(treebank_tag: str) -> str:
    """Map a Penn Treebank tag (NN, VBD, JJ, RB ...) to the WordNet POS the lemmatizer expects."""
    return {"J": wordnet.ADJ, "V": wordnet.VERB, "R": wordnet.ADV}.get(treebank_tag[:1], wordnet.NOUN)


def remove_non_letters(text: str) -> str:
    """Lower-case and replace every character that is not a-z (digits, punctuation, symbols) with a space."""
    return NON_LETTERS.sub(" ", str(text).lower()).strip()


def tokenize(text: str) -> list[str]:
    ensure_nltk_data()
    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    sw = stop_words()
    return [t for t in tokens if t not in sw and len(t) >= MIN_TOKEN_LEN]


def lemmatize(tokens: list[str], use_pos: bool = True) -> list[str]:
    """Reduce each token to its dictionary form. With `use_pos`, verbs are lemmatized as verbs
    (said -> say, rising -> rise) instead of being treated as nouns, which is WordNet's default."""
    lem = _lemmatizer()
    if not use_pos:
        return [t if KEEP_AS_IS.search(t) else lem.lemmatize(t) for t in tokens]
    return [t if KEEP_AS_IS.search(t) else lem.lemmatize(t, _wordnet_pos(tag)) for t, tag in nltk.pos_tag(tokens)]


def clean_text(text: str, use_pos: bool = True) -> str:
    """Full pipeline used for modelling. Returns the processed article as one space-separated string."""
    tokens = tokenize(remove_non_letters(text))
    tokens = remove_stopwords(tokens)
    return " ".join(lemmatize(tokens, use_pos=use_pos))


def process_steps(text: str, use_pos: bool = True) -> dict[str, object]:
    """The intermediate output of every step, for showing how one article changes."""
    letters = remove_non_letters(text)
    tokens = tokenize(letters)
    no_stop = remove_stopwords(tokens)
    lemmas = lemmatize(no_stop, use_pos=use_pos)
    return {"original": text, "letters_only": letters, "tokens": tokens,
            "without_stopwords": no_stop, "lemmatized": lemmas, "final": " ".join(lemmas)}
