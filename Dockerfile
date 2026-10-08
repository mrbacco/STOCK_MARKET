# syntax=docker/dockerfile:1.7

# One immutable image serves the Streamlit replicas and both worker roles.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app
RUN groupadd --system app && useradd --system --gid app --create-home app

# LightGBM's Linux wheel links against the OpenMP runtime, which the slim
# base image does not include.
RUN apt-get update     && apt-get install -y --no-install-recommends libgomp1     && rm -rf /var/lib/apt/lists/*

# The lockfile pins every transitive dependency and selects the CPU-only
# PyTorch wheel, so the image does not download CUDA libraries it never uses.
COPY requirements.lock ./
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.lock

COPY . .
RUN mkdir -p /app/data && chown -R app:app /app

USER app
EXPOSE 8501

# This endpoint does not execute the market or model pipelines.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

CMD ["python", "-m", "streamlit", "run", "app.py"]
