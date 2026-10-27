**Problem.** FlipItNews needs every incoming article filed under the right section (Business, Entertainment,
Politics, Sports, Technology) so it can be recommended, searched and grouped. The data is **2,225 labelled news
articles** (2,126 unique), with Sports (511) and Business (510) the largest categories and Technology (401) and
Entertainment (386) the smallest.

**Approach.** A user-defined function cleans every article in four steps: remove non-letters, word-tokenise,
remove stopwords, and POS-aware WordNet lemmatisation. That removes 44% of tokens and halves the vocabulary. The
target is label-encoded and the user chooses **Bag of Words or TF-IDF** (unigrams + bigrams, 5,000 features,
fitted on the training data only). After a stratified **75:25 split (1,668 / 557 articles)**, Naive Bayes is trained
the simple way, then a functionalised routine trains and evaluates Decision Tree, Nearest Neighbors and Random
Forest, plus Logistic Regression and a Linear SVM as references, with both vectorizers. The ranking is checked with
5-fold cross-validation, a duplicate-leakage test, error analysis and a confidence analysis.

**Results.**

| Model (TF-IDF) | Test accuracy | 5-fold CV accuracy | Comment |
|---|---|---|---|
| **Logistic Regression** (deployed) | **98.0%** | **97.8%** | Best in CV, calibrated probabilities, 0.4 MB |
| Linear SVM | 98.2% | 97.7% | Tied with Logistic Regression, but no probabilities |
| **Naive Bayes** | 97.1% (97.7% BoW) | 97.2% | **Best of the four required models**, trains in milliseconds |
| Nearest Neighbors | 97.3% | 96.3% | Needs TF-IDF (94.3% with BoW); stores the whole training set |
| Random Forest | 97.0% | 95.9% | Robust but ~15 MB and the slowest |
| Decision Tree | 85.6% | 80.6% | Over-fits; not suitable on its own |

**Key findings.** (1) TF-IDF is the better representation: +3.1 points for Nearest Neighbors and about +1 point for
the linear models, at no extra cost. (2) Removing the 99 duplicate articles does not reduce the best models'
accuracy, so the results are genuine. (3) The 11 remaining errors are genuinely cross-topic stories, and Business,
which overlaps with every other section, is involved in 7 of them. (4) Sending articles with confidence below 0.6
to an editor (4.5% of articles) catches 10 of the 11 errors and leaves **99.8% accuracy** on the rest.

**Recommendation.** Deploy Logistic Regression + TF-IDF with the 0.6 confidence rule, keep Naive Bayes as a
lightweight fallback, and retrain on FlipItNews's own (Indian finance) articles, because some of the strongest
features are 2004-05 British names. The model is served through a FastAPI REST API and a Streamlit web app,
packaged with Docker and tested in CI.
