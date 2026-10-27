"""FlipItNews: news article categorisation with classical NLP (BoW / TF-IDF + scikit-learn)."""
from .preprocess import clean_text, process_steps

__version__ = "1.0.0"
__all__ = ["clean_text", "process_steps"]
