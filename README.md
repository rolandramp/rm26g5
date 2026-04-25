# rm26g5
Research Methods 2026 Group 5

## Parametrizable RAG Evaluation Framework

A Python 3.14 project for evaluating Retrieval-Augmented Generation (RAG) pipelines against ground-truth datasets.

### Features

- Configurable document loaders, chunking strategies, embedding models, vector databases, retrievers, and LLMs
- Support for OpenAI-compatible REST APIs (including local Ollama instances)
- Local persistent vector storage with Chroma
- Integration with huggingface datasets (rag-mini-bioasq benchmark)
- Multiple embedding profile support (nomic-embed-text, mxbai-embed-large, bge-m3)
- Multiple LLM profile support (ministral-3:3b, gpt-3.5-turbo)
- CLI and FastAPI for parameterized experiments
- Result persistence and metadata logging

### Installation

1. Create Conda environment:
```bash
conda env create -f environment.yml
conda activate rag-env
```

2. Install the package:
```bash
pip install -e .
```

### Configuration

Configure experiment in `config/experiment_config.yaml`:

- **Ingestion**: Dataset source and cache directories
- **Chunking**: Strategy, chunk size, and overlap
- **Embedding**: Multiple embedding profiles with different models and APIs
- **Vector Store**: Store type and persistence directory
- **Retriever**: Number of documents to retrieve (k)
- **Generator**: Multiple LLM profiles with different models and APIs
- **Evaluation**: Evaluation metric configuration

### Usage

#### CLI

Run experiment:
```bash
rag-eval run-experiment --config config/experiment_config.yaml --output results/
```

#### API Server

Start the FastAPI server:
```bash
python -m rag_framework.api
```

Or with uvicorn:
```bash
uvicorn rag_framework.api:app --host 0.0.0.0 --port 8000
```

##### Endpoints

- `GET /` - Service info
- `GET /health` - Health check
- `POST /run-experiment` - Run experiment

Example request:
```bash
curl -X POST http://localhost:8000/run-experiment \
  -H "Content-Type: application/json" \
  -d '{"config_path": "config/experiment_config.yaml", "output_dir": "./results"}'
```

### Project Structure

- `src/rag_framework/` - Core modules
  - `api.py` - FastAPI server
  - `chunking.py` - Document chunking strategies
  - `embedding.py` - Embedding model integration
  - `evaluation.py` - Evaluation metrics
  - `generator.py` - LLM generation
  - `ingestion.py` - Dataset loading
  - `main.py` - CLI entry point
  - `retriever.py` - Retrieval logic
  - `vector_store.py` - Vector database integration
- `config/` - Configuration files
- `data/` - Input datasets
- `cache/` - Dataset cache
- `chroma_db/` - Vector database persistence
- `results/` - Experiment results
- `dockerimage/ollama/` - Ollama Docker configuration

### Docker

Start all services with Docker Compose:
```bash
docker-compose up -d
```

This starts:
- **ollama-backend** - Ollama LLM service (internal network)
- **rag-framework** - FastAPI server on port 8000

Volume mounts:
- `./src` → `/app/src`
- `./config` → `/app/config`
- `./data` → `/app/data`
- `./cache` → `/app/cache`
- `./chroma_db` → `/app/chroma_db`
- `./results` → `/app/results`

### Dependencies

Managed via `pyproject.toml` (PEP 621). No separate requirements.txt.