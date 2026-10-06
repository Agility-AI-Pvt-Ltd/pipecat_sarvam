FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY pipecat_sarvam_vobiz ./pipecat_sarvam_vobiz

RUN pip install --upgrade pip \
    && pip install .

# Pipecat splits the LLM's reply into sentences before sending it to TTS, using
# NLTK's punkt_tab data. Without it that split raises and the bot never speaks.
# Pipecat downloads it on import otherwise — on every container start, since the
# container filesystem doesn't persist — so a network blip at boot would leave a
# container that answers calls in silence. Baked in here, on NLTK's default path.
RUN pip install nltk \
    && python -c "import nltk; assert nltk.download('punkt_tab', download_dir='/usr/local/share/nltk_data', quiet=True)"

EXPOSE 8000

CMD ["uvicorn", "pipecat_sarvam_vobiz.main:app", "--host", "0.0.0.0", "--port", "8000"]
