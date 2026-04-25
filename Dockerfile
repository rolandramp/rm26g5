FROM python:3.14-slim

WORKDIR /app

RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/
COPY pyproject.toml .
COPY environment.yml .

RUN pip install -e .

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["python", "-m", "rag_framework.api"]