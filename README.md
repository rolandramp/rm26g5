# RAG Evaluation Framework 

This is the project of group 5 of the lecture research methods 2026.

A Python-based framework for evaluating Retrieval-Augmented Generation (RAG) pipelines against ground-truth datasets. Designed for researchers to systematically compare different embedding models, chunking strategies, and LLM configurations.

## Features

- **Configurable Pipelines**: Easily swap embedding models, chunking strategies, vector stores, and LLMs
- **Multi-Backend Support**: Works with Ollama, llama.cpp, and OpenAI-compatible APIs
- **Comprehensive Metrics**: Uses RAGAS for semantic similarity, answer correctness, context precision/recall, and faithfulness
- **CLI & API**: Run experiments from command line or via REST API
- **Docker-Ready**: Full containerized setup with Docker Compose

## Architecture

```
                    ┌─────────────────────┐
                    │  Config (YAML)      │
                    └─────────┬──────────┘
                              │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
     │ Ingestion  │→ │ Chunking    │→ │ Embedding   │
     └─────────────┘  └─────────────┘  └─────────────┘
              │              │              │
              ▼              │              ▼
     ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
     │ Vector Store│← │ Retriever   │  │ LLM (Judge) │
     │  (Chroma)   │  └─────────────┘  └─────────────┘
     └─────────────┘        │
              │            ▼
              │    ┌─────────────┐
              │    │ Generator  │
              │    └─────────────┘
              │            │
              ▼            ▼
        ┌─────────────────────────────┐
        │     RAGEvaluator            │
        │ (RAGAS + Custom Metrics)   │
        └─────────────────────────────┘
```

## Prerequisites

- Python 3.12+
- Docker & Docker Compose
- GPU (optional, for llama.cpp)

## Quick Start

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# Start all services with GPU
docker-compose --profile gpu up -d

# Run experiment via CLI
docker exec rag-framework rag-eval run-experiment \
  --config config/experiment_config.yaml \
  --output results/my-experiment

# Or via API
curl -X POST http://localhost:8000/run-experiment \
  -H "Content-Type: application/json" \
  -d '{"config_path": "config/experiment_config.yaml", "output_dir": "./results"}'
```

### Local Development

```bash
# Create environment
conda env create -f environment.yml
conda activate rag-env

# Install package
pip install -e .

# Run experiment
rag-eval run-experiment --config config/experiment_config.yaml --output results/
```

## Running Experiments

### CLI

```bash
rag-eval run-experiment \
  --config config/experiment_config.yaml \
  --output results/my-experiment
```

### API Server

Start the server:

```bash
# Direct
python -m rag_framework.api

# Or with uvicorn
uvicorn rag_framework.api:app --host 0.0.0.0 --port 8000
```

API Endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| POST | `/run-experiment` | Run experiment |

Example:

```bash
curl -X POST http://localhost:8000/run-experiment \
  -H "Content-Type: application/json" \
  -d '{
    "config_path": "config/experiment_config.yaml",
    "output_dir": "./results"
  }'
```

## Configuration

The experiment is configured via `config/experiment_config.yaml`:

```yaml
# Data ingestion
ingestion:
  dataset: rag-datasets/rag-mini-bioasq
  dataset_qna_config: question-answer-passages
  dataset_corpus_config: text-corpus
  dataset_cache_dir: ./cache/datasets

# Chunking strategy
chunking:
  strategy: recursive
  chunk_size: 1000
  chunk_overlap: 200

# Embedding models (multiple profiles)
active_embedding_profile: nomic-embed-text
embedding_profiles:
  nomic-embed-text:
    model_name: nomic-embed-text
    api_key: ollama
    base_url: http://ollama-backend:11434/v1

# Generator models (multiple profiles)
active_generator_profile: gpt-3.5-turbo
generator_profiles:
  mistral-3-3b-gguf:
    model_name: mistralai/Ministral-3-3B-Instruct-2512-GGUF:Q4_K_M
    base_url: http://llama-cpp-backend:8080/v1
  gpt-3.5-turbo:
    model_name: openai/gpt-3.5-turbo
    base_url: https://openrouter.ai/api/v1/

# Workload limits
workload:
  max_chunks: 100
  max_questions: 10
```

## Project Structure

```
.
├── src/rag_framework/       # Core package
│   ├── api.py              # FastAPI server
│   ├── chunking.py         # Document chunking
│   ├── embedding.py         # Embedding models
│   ├── evaluation.py       # RAGAS + custom metrics
│   ├── generator.py        # LLM generation
│   ├── ingestion.py       # Dataset loading
│   ├── main.py           # CLI entry point
│   ├── retriever.py      # Retrieval logic
│   ├── service.py         # Experiment service
│   └── vector_store.py   # Chroma integration
├── config/
│   └── experiment_config.yaml
├── dockerimage/
│   ├── llama-cpp/         # llama.cpp Docker
│   └── ollama/            # Ollama Docker
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── environment.yml
```

## Docker Services

| Service | Port | Description | GPU |
|---------|------|-------------|-----|
| rag-framework | 8000 | FastAPI server | - |
| ollama-backend | 6543 | Ollama with embeddings | Optional |
| llama-cpp-backend | 1235 | llama.cpp with GGUF models | Required |


## Technology Stack

- **LangChain**: RAG pipeline components
- **Chroma**: Vector storage
- **RAGAS**: Evaluation metrics
- **FastAPI**: REST API
- **Ollama**: Local LLM/embeddings
- **llama.cpp**: GGUF model inference

## License

See [LICENSE](./LICENSE) file.