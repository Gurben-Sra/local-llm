FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" \
    GRADIO_ANALYTICS_ENABLED=False

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libopenblas-dev \
        libgomp1 \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY run.py server.py ui.py ./
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /models \
    && chown app:app /app /models

USER app
EXPOSE 8000 7860

CMD ["python", "server.py"]
