FROM python:3.14-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/

# Run as a non-root user: the pipeline only ever reads an input image and calls out to models
# already on disk/the Hub, so there's no reason for it to run as root inside the container.
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["python", "-m", "src.pipeline"]
