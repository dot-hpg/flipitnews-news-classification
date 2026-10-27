from flipitnews.preprocess import (clean_text, lemmatize, process_steps, remove_non_letters,
                                   remove_stopwords, tokenize)


def test_remove_non_letters_strips_digits_punctuation_and_case():
    assert remove_non_letters("Shares ROSE 5.2% to £11bn!") == "shares rose to bn"


def test_tokenize_splits_words():
    assert tokenize("the bank cut rates") == ["the", "bank", "cut", "rates"]


def test_remove_stopwords_drops_function_words_and_single_letters():
    assert remove_stopwords(["the", "bank", "of", "s", "england"]) == ["bank", "england"]


def test_lemmatize_uses_part_of_speech():
    assert lemmatize(["prosecutors", "said", "shares", "rising"]) == ["prosecutor", "say", "share", "rise"]


def test_lemmatize_keeps_words_wordnet_would_break():
    assert lemmatize(["boss", "us", "virus"]) == ["boss", "us", "virus"]


def test_clean_text_end_to_end():
    assert clean_text("Profits ROSE 12% in 2005!") == clean_text("profits rose in")
    out = clean_text("The striker scored twice as Arsenal won the match.")
    assert out.split() == ["striker", "score", "twice", "arsenal", "win", "match"]


def test_clean_text_handles_empty_and_non_text():
    assert clean_text("") == ""
    assert clean_text("123 !!! 456") == ""


def test_process_steps_exposes_every_stage():
    steps = process_steps("The markets were rising.")
    assert steps["letters_only"] == "the markets were rising"
    assert steps["without_stopwords"] == ["markets", "rising"]
    assert steps["final"] == "market rise"
