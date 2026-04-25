FROM python:3.14-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY environment.yml .
RUN pip install -e .

COPY src/ ./src/
COPY config/ ./config/

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["python", "-m", "rag_framework.api"]