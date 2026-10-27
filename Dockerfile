# FlipItNews category classifier: one image, two entry points
#   API (default):  docker run -p 8000:8000 flipitnews
#   Web app:        docker run -p 8501:8501 flipitnews streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

FROM python:3.11-slim AS deps
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.txt /tmp/requirements.txt
RUN pip install --prefix=/install -r /tmp/requirements.txt

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 NLTK_DATA=/usr/share/nltk_data
COPY --from=deps /install /usr/local
RUN python -m nltk.downloader -d /usr/share/nltk_data stopwords punkt_tab wordnet averaged_perceptron_tagger_eng \
    && useradd --create-home --uid 10001 app
WORKDIR /app
COPY flipitnews/ flipitnews/
COPY api/ api/
COPY streamlit_app.py ./
COPY models/ models/
COPY reports/figures/ reports/figures/
USER app
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
