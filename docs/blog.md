# Sorting the News with Classical NLP: 98% Accuracy Without a Transformer

*How I built a news-category classifier for a fintech content platform: what mattered, what didn't, and two
small text-cleaning bugs that most tutorials would ship.*

---

## The problem

FlipItNews is a Gurugram-based platform that wants to make finance and markets easier to follow for first-time
investors. Behind the recommendations, search and "related stories" panels sits a simple question: **which section
does this article belong to?** Business, Politics, Sports, Technology or Entertainment.

I had 2,225 labelled articles to work with. The brief asked for classical NLP (stopwords, tokenisation,
lemmatisation, Bag of Words, TF-IDF) and a comparison of at least three classifiers. I wanted to go further and
answer the questions a product team would actually ask: *How much can we trust it? Which errors will it make? What
happens when we deploy it?*

## Step 1: Look at the data before modelling

Three things stood out:

* **The classes are balanced enough.** Sports (511) and Business (510) lead, Entertainment (386) is the smallest.
  The largest class is only 1.3x the smallest, so accuracy is meaningful, but I reported macro F1 next to it anyway.
* **99 articles are exact duplicates.** Same text, same label. Harmless as label noise, but a leakage risk: one
  copy can end up in training and its twin in the test set. I kept them for the main analysis (the brief describes
  2,225 articles) and measured their effect separately.
* **Each category has its own vocabulary.** Sports talks about *game, win, player*; Politics about *labour,
  government, election*; Business about *company, market, bn*. A bag-of-words approach should work well here.

## Step 2: Clean the text, and test the cleaning

The cleaning function does four things: remove non-letters, tokenise, remove stopwords, and lemmatise using each
word's part of speech, so *said* becomes *say* and *rising* becomes *rise*. That cut 44% of tokens and halved the
vocabulary.

Writing unit tests for this "boring" step caught two bugs I would otherwise have shipped:

1. **NLTK's stopword list contains `won`.** It is there as the stem of *won't*, but in news it is almost always
   the past tense of *win*. "Arsenal won the match" was being reduced to "arsenal match", losing the most useful
   Sports word in the sentence.
2. **WordNet thinks `boss` is the plural of `bos`** (the cattle genus) and `us` is the plural of `u`. So
   "WorldCom boss" became "worldcom bos". Words ending in *-ss* and *-us* are now left alone.

Neither changes accuracy dramatically, but both are visible to anyone who reads the model's explanations. If the
top words are going to be shown to editors, they have to be real words.

## Step 3: Bag of Words vs TF-IDF, with a fair comparison

The user can switch between the two representations, and both use exactly the same vocabulary (unigrams + bigrams,
top 5,000, fitted on the training split only, so no test information leaks into the IDF weights).

| Model | Bag of Words | TF-IDF |
|---|---|---|
| Logistic Regression | 97.0% | **98.0%** |
| Linear SVM | 97.0% | 98.2% |
| Naive Bayes | 97.7% | 97.1% |
| Nearest Neighbors | 94.3% | 97.3% |
| Random Forest | 97.0% | 97.0% |
| Decision Tree | 83.8% | 85.6% |

TF-IDF helps the models that measure similarity or learn weights: **+3.1 points for Nearest Neighbors**, since raw
counts make every article look similar through words like *say* and *year*. Random Forest doesn't care, because a
tree splits one feature at a time. Naive Bayes, which is defined on counts, is a tie within noise.

## Step 4: Don't trust a single split

With 557 test articles, one error is worth 0.18 points. So I re-ranked everything with 5-fold cross-validation on
the training set:

* **Logistic Regression (97.8%) ≈ Linear SVM (97.7%) ≈ Naive Bayes (97.2%)**, all within about one standard deviation
* Nearest Neighbors 96.3%, Random Forest 96.1%
* **Decision Tree 80.6-83.0%.** Its single-split score of 85.6% was flattering.

Among the four models the brief required, **Naive Bayes wins**: the best cross-validated accuracy, it trains in
about 5 milliseconds, and you can read its reasoning word by word.

## Step 5: Were the duplicates inflating the score?

42 of the 557 test articles had an exact copy in training. I re-ran the whole comparison on the 2,126 unique
articles:

* Naive Bayes, Logistic Regression and the SVM: **unchanged** (within ±0.2 points)
* Decision Tree −2.0, Random Forest −1.1, Nearest Neighbors −0.5

The models that *memorise* lost ground, and the ones that *generalise* did not. That is a reassuring result, and
it took two minutes to check.

## Step 6: Read the errors

The deployed Logistic Regression makes 11 mistakes out of 557. Reading them is the best part of the project,
because nearly all of them are stories a human editor could also file in two places:

* McDonald's sponsoring an MTV show (labelled Business, predicted Entertainment)
* MPs threatened with expulsion for using BlackBerrys in the Commons (Politics → Technology). This one appears
  twice, because it is one of the duplicates.
* A UK state-pension overhaul (Business → Politics)
* PC ownership forecasts (Technology → Business)

Business is involved in 7 of the 11 errors, and Sports in none. Business is the "hub" category that overlaps with
everything else.

## Step 7: Make it useful, not just accurate

Logistic Regression returns probabilities, so I asked: *if low-confidence articles go to a human, what happens?*

| Confidence threshold | Auto-filed | Accuracy on auto-filed | Errors caught |
|---|---|---|---|
| none | 100% | 98.0% | 0 of 11 |
| **0.6** | **95.5%** | **99.8%** | **10 of 11** |
| 0.8 | 86.5% | 100.0% | 11 of 11 |

Reviewing one article in twenty turns a 98% model into a near error-free workflow. That rule is built into the web
app.

## The honest caveat

The model's most indicative words include *Yukos*, *Mr Blair* and *Michael Howard*. This is British news from
2004-05. When I gave it an Indian cricket report, it said Sports, but at only 51% confidence, because *century*
and *wicket* are rare in the training data. The confidence rule catches exactly this kind of drift, and the first
production step would be retraining on FlipItNews's own articles.

## What I shipped

* A reusable Python package (`flipitnews/`) that the notebook, the training script, the API and the app all import,
  so the text is processed identically everywhere
* A **FastAPI** service (`/predict` with model and vectorizer choice, probabilities and top words)
* A **Streamlit** app: paste an article, toggle BoW / TF-IDF, see the category and the words behind it
* **Docker** + docker-compose, **28 tests**, and **GitHub Actions** CI that retrains from the raw CSV, runs the
  tests, builds the image and smoke-tests the container

## Takeaways

1. **Classical NLP is still a strong baseline.** 98% with a 0.4 MB model that answers in under a millisecond.
2. **Test your preprocessing.** Two library defaults were quietly damaging the text.
3. **Rank models with cross-validation**, not one split, and check for duplicate leakage.
4. **Probabilities are a product feature.** A confidence threshold plus human review beats chasing the last point
   of accuracy.

*Code, notebook and report: [github.com/dot-hpg/flipitnews-news-classification](https://github.com/dot-hpg/flipitnews-news-classification)*

*Guntuku Hariprasad*
