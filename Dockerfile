FROM python:3.12-slim

WORKDIR /app

RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ ./src/
COPY config/ ./config/

RUN pip install --no-cache-dir -e .

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["python", "-m", "rag_framework.api"]