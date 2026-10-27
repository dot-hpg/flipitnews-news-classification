"""FlipItNews article classifier: paste an article, pick BoW or TF-IDF and a model, see the category and why.

    streamlit run streamlit_app.py
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from flipitnews.inference import NewsClassifier
from flipitnews.preprocess import ensure_nltk_data, process_steps

ROOT = Path(__file__).resolve().parent
COLORS = {"Business": "#2a78d6", "Entertainment": "#e8590c", "Politics": "#7048e8",
          "Sports": "#1baf7a", "Technology": "#d6336c"}
MODEL_LABELS = {"logistic_regression": "Logistic Regression (deployed)", "naive_bayes": "Naive Bayes",
                "nearest_neighbors": "Nearest Neighbors", "random_forest": "Random Forest",
                "decision_tree": "Decision Tree"}
SAMPLES = {
    "Markets": "Shares in Asian markets rose on Monday after the central bank held interest rates steady and "
               "signalled that inflation was easing. Banks led the gains as investors bet on stronger profits, "
               "while the rupee firmed against the dollar.",
    "Cricket": "The captain struck an unbeaten century as India chased down 320 to win the third one-day "
               "international and clinch the series 2-1. The bowlers had earlier restricted the visitors with "
               "three late wickets.",
    "Gadgets": "The company unveiled a new smartphone with a faster processor, an improved camera and software "
               "that uses artificial intelligence to edit photos. Analysts said broadband and mobile users would "
               "drive demand.",
    "Parliament": "The opposition walked out of parliament after the minister refused to answer questions on the "
                  "election bill. Party leaders accused the government of rushing the law through before the vote.",
    "Cinema": "The film won best picture at the awards ceremony, and its director thanked the cast and the music "
              "composer. The lead actress also took home the top acting prize after a record box office run.",
}

st.set_page_config(page_title="FlipItNews Classifier", page_icon="📰", layout="wide")


@st.cache_resource
def load() -> NewsClassifier:
    ensure_nltk_data()
    return NewsClassifier.load()


clf = load()

st.title("📰 FlipItNews article classifier")
st.caption("Classical NLP (lemmatisation + Bag of Words / TF-IDF) and scikit-learn, trained on 2,126 unique news "
           "articles in five categories. 98% accuracy on held-out articles with the deployed model.")

tab_one, tab_batch, tab_models = st.tabs(["Classify an article", "Batch (CSV)", "Model comparison"])

with st.sidebar:
    st.header("Settings")
    vec_label = st.radio("Text representation", ["TF-IDF", "Bag of Words"],
                         help="TF-IDF down-weights words that appear in every article; Bag of Words uses raw counts.")
    vectorizer = "tfidf" if vec_label == "TF-IDF" else "bow"
    model = st.selectbox("Model", list(MODEL_LABELS), format_func=MODEL_LABELS.get)
    m = clf.metrics.get(f"{model}__{vectorizer}", {})
    if m:
        st.metric("Held-out accuracy", f"{m['accuracy']:.1%}", help="Stratified 25% test split, 557 articles")
        st.metric("Macro F1", f"{m['f1_macro']:.3f}")
    st.divider()
    st.markdown("[Source on GitHub](https://github.com/dot-hpg/flipitnews-news-classification)")

with tab_one:
    pick = st.pills("Try a sample", list(SAMPLES), selection_mode="single")
    text = st.text_area("Article text", value=SAMPLES.get(pick, ""), height=190,
                        placeholder="Paste a news article or headline here ...")
    if st.button("Classify", type="primary", disabled=not text.strip()):
        res = clf.predict([text], model, vectorizer)[0]
        cat, conf = res["category"], res["confidence"]
        c1, c2 = st.columns([1, 1.4])
        with c1:
            st.markdown(f"<div style='padding:18px;border-radius:12px;background:{COLORS[cat]};color:white'>"
                        f"<div style='font-size:13px;opacity:.85'>Predicted category</div>"
                        f"<div style='font-size:34px;font-weight:700'>{cat}</div>"
                        f"<div style='font-size:15px'>{conf:.0%} confidence</div></div>", unsafe_allow_html=True)
            if conf < 0.6:
                st.warning("Low confidence: this article may belong to two sections. Send it to an editor.")
            if res["top_terms"]:
                st.markdown("**Words that drove the decision**")
                st.markdown(" ".join(f"`{t['term']}`" for t in res["top_terms"]))
            elif res["tokens_used"] == 0:
                st.info("None of the article's words are in the model's vocabulary.")
        with c2:
            probs = pd.Series(res["probabilities"]).sort_values()
            st.markdown("**Probability by category**")
            st.bar_chart(probs.rename("probability"), horizontal=True, color="#2a78d6", height=230)
        with st.expander("See how the text was processed"):
            steps = process_steps(text)
            st.markdown(f"**1. Non-letters removed:** {steps['letters_only'][:600]}")
            st.markdown(f"**2. Tokens ({len(steps['tokens'])}):** {', '.join(steps['tokens'][:60])}")
            st.markdown(f"**3. Stopwords removed ({len(steps['without_stopwords'])}):** "
                        f"{', '.join(steps['without_stopwords'][:60])}")
            st.markdown(f"**4. Lemmatised:** {steps['final'][:600]}")

with tab_batch:
    st.write("Upload a CSV with an **Article** column (or paste one article per line). Up to 500 rows.")
    up = st.file_uploader("CSV file", type="csv")
    pasted = st.text_area("... or one article per line", height=120)
    rows = []
    if up is not None:
        df_up = pd.read_csv(up)
        col = next((c for c in df_up.columns if c.lower() in ("article", "text", "content")), None)
        if col is None:
            st.error("No 'Article' column found.")
        else:
            rows = df_up[col].dropna().astype(str).tolist()[:500]
    elif pasted.strip():
        rows = [r for r in pasted.splitlines() if r.strip()][:500]
    if rows:
        preds = clf.predict(rows, model, vectorizer)
        out = pd.DataFrame({"article": [r[:120] for r in rows], "category": [p["category"] for p in preds],
                            "confidence": [p["confidence"] for p in preds]})
        st.dataframe(out, use_container_width=True, hide_index=True)
        st.bar_chart(out.category.value_counts())
        buf = io.StringIO()
        out.to_csv(buf, index=False)
        st.download_button("Download predictions", buf.getvalue(), "flipitnews_predictions.csv", "text/csv")

with tab_models:
    mt = pd.DataFrame(clf.metrics.values())
    mt["vectorizer"] = mt.vectorizer.map({"bow": "Bag of Words", "tfidf": "TF-IDF"})
    st.subheader("Held-out test set (557 articles)")
    st.dataframe(mt.sort_values("accuracy", ascending=False), hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(format="%.3f")
                                for c in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]})
    figs = ROOT / "reports" / "figures"
    for name, cap in [("06_bow_vs_tfidf", "Bag of Words vs TF-IDF"),
                      ("05_confusion_matrices_tfidf", "Confusion matrices (TF-IDF)"),
                      ("07_indicative_words", "Most indicative words per category")]:
        if (figs / f"{name}.png").exists():
            st.image(str(figs / f"{name}.png"), caption=cap)
    card = ROOT / "models" / "model_card.json"
    if card.exists():
        with st.expander("Model card"):
            st.json(json.loads(card.read_text()))
