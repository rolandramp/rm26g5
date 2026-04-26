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
  --config config/experiment_config_docker.yaml \
  --output results/my-experiment

# Or via API
curl -X POST http://localhost:8000/run-experiment \
  -H "Content-Type: application/json" \
  -d '{"config_path": "config/experiment_config_docker.yaml", "output_dir": "./results"}'
```

### Local Development

```bash
# Create environment
conda env create -f environment.yml
conda activate rag-env

# Install package
pip install -e .

# Run experiment
rag-eval run-experiment --config config/experiment_config_local.yaml --output results/
```

## Running Experiments

### CLI

```bash
# Run single experiment
rag-eval run-experiment \
  --config config/experiment_config_local.yaml \
  --output results/my-experiment

# Run all combinations (embedding x generator x chunking profiles)
rag-eval run-experiment \
  --config config/experiment_config_local.yaml \
  --output results/my-experiment \
  --run-all
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
| POST | `/run-experiment` | Run experiment (single or all combinations) |

Example:

```bash
# Single experiment
curl -X POST http://localhost:8000/run-experiment \
  -H "Content-Type: application/json" \
  -d '{
    "config_path": "config/experiment_config_local.yaml",
    "output_dir": "./results"
  }'

# Run all combinations (embedding x generator x chunking profiles)
curl -X POST http://localhost:8000/run-experiment \
  -H "Content-Type: application/json" \
  -d '{
    "config_path": "config/experiment_config_local.yaml",
    "output_dir": "./results",
    "run_all": true
  }'
```

PowerShell:

```powershell
# Single experiment
Invoke-RestMethod -Uri "http://localhost:8000/run-experiment" -Method Post -ContentType "application/json" -Body '{"config_path":"config/experiment_config_local.yaml","output_dir":"./results"}'

# Run all combinations
Invoke-RestMethod -Uri "http://localhost:8000/run-experiment" -Method Post -ContentType "application/json" -Body '{"config_path":"config/experiment_config_local.yaml","output_dir":"./results","run_all":true}'
```

## Configuration

The experiment is configured via `config/experiment_config_local.yaml` (or `experiment_config_docker.yaml` for Docker):

```yaml
# Data ingestion
ingestion:
  dataset: rag-datasets/rag-mini-bioasq
  dataset_qna_config: question-answer-passages
  dataset_corpus_config: text-corpus
  dataset_cache_dir: ./cache/datasets

# Chunking strategy (multiple profiles)
active_chunking_profile: medium
chunking_profiles:
  small:
    strategy: recursive
    chunk_size: 500
    chunk_overlap: 100
  medium:
    strategy: recursive
    chunk_size: 1000
    chunk_overlap: 200
  large:
    strategy: recursive
    chunk_size: 2000
    chunk_overlap: 400

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

## Evaluation Metrics

This framework uses RAGAS (Retrieval Augmented Generation Assessment) for RAG pipeline evaluation, supplemented with traditional NLP metrics. Below is a detailed description of each metric used:

### RAGAS Metrics

#### Faithfulness

**Purpose**: Measures how factually consistent the generated answer is with the retrieved context.

**Range**: 0 to 1 (higher is better)

**How it works**:
1. Break the generated answer into individual statements/claims
2. For each statement, verify if it can be inferred from the retrieved context
3. Calculate: `Faithfulness = Supported Claims / Total Claims`

**Interpretation**: A score of 1.0 means all claims in the answer are supported by the context. A score of 0.5 means only half the claims are supported - the model may be hallucinating or making unsupported statements.

**Example**:
- Context: "Albert Einstein was born on 14 March 1879 in Germany."
- High faithfulness (1.0): "Einstein was born in Germany on 14 March 1879."
- Low faithfulness (0.5): "Einstein was born in Germany on 20 March 1879."

---

#### Semantic Similarity

**Purpose**: Evaluates the semantic resemblance between the generated answer and the ground truth answer.

**Range**: 0 to 1 (higher is better)

**How it works**:
1. Vectorize the reference answer using an embedding model
2. Vectorize the generated response using the same embedding model
3. Compute cosine similarity between the two vectors

**Interpretation**: Measures whether the generated answer captures the same meaning as the ground truth, even if the exact words differ. More robust than exact match since it captures semantic similarity.

---

#### Answer Correctness

**Purpose**: Assesses the overall accuracy of the generated answer compared to the ground truth.

**Range**: 0 to 1 (higher is better)

**How it works**: Combines two aspects:
1. **Factual Correctness**: Uses F1 score to measure statement overlap between generated and ground truth (TP/FP/FN)
2. **Semantic Similarity**: Embedding-based similarity (as described above)
3. **Weighted Average**: Combines both scores (default weights: 75% factual, 25% semantic)

**Interpretation**: A comprehensive measure that catches both factual errors and semantic drift. Lower scores indicate the answer contains incorrect information or diverges significantly from the expected answer.

---

#### Context Precision

**Purpose**: Evaluates the retriever's ability to rank relevant chunks higher than irrelevant ones.

**Range**: 0 to 1 (higher is better)

**How it works**:
1. For each retrieved chunk at position k, determine if it's relevant to answer the question
2. Calculate precision@k: `TP@k / (TP@k + FP@k)`
3. Weight by relevance and compute mean across all chunks

**Interpretation**: High precision means relevant information appears at the top of the retrieval results. Low precision indicates the retriever ranks irrelevant chunks too high, making it harder for the LLM to find the right answer.

---

### Traditional NLP Metrics

#### BLEU Score (Bilingual Evaluation Understudy)

**Purpose**: Measures n-gram precision between generated text and reference text.

**Range**: 0 to 1 (higher is better)

**How it works**:
1. Count matching n-grams (unigrams, bigrams, trigrams, 4-grams) between generated and reference
2. Calculate precision for each n-gram level
3. Apply brevity penalty to penalize short outputs
4. Compute geometric mean of precisions

**Strengths**: Simple, fast, widely used for machine translation
**Limitations**: Does not capture semantic meaning, only lexical overlap

---

#### ROUGE Score (Recall-Oriented Understudy for Gisting Evaluation)

**Purpose**: Measures recall-oriented n-gram overlap between generated and reference text.

**Range**: 0 to 1 (higher is better)

**How it works**:
- **ROUGE-N**: Counts overlapping n-grams (unigrams, bigrams, etc.)
- **ROUGE-L**: Uses Longest Common Subsequence to capture structural similarity

**Formula**: `ROUGE = Overlapping n-grams in generated text / Total n-grams in reference`

**Strengths**: Captures content coverage, good for summarization tasks
**Limitations**: Does not consider semantic equivalence, only lexical overlap

---

### Metric Interpretation Guide

| Metric | Poor | Acceptable | Good | Excellent |
|--------|------|------------|------|-----------|
| Faithfulness | < 0.5 | 0.5 - 0.7 | 0.7 - 0.9 | > 0.9 |
| Semantic Similarity | < 0.6 | 0.6 - 0.75 | 0.75 - 0.9 | > 0.9 |
| Answer Correctness | < 0.5 | 0.5 - 0.7 | 0.7 - 0.85 | > 0.85 |
| Context Precision | < 0.5 | 0.5 - 0.7 | 0.7 - 0.85 | > 0.85 |
| BLEU/ROUGE | < 0.3 | 0.3 - 0.5 | 0.5 - 0.7 | > 0.7 |

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
│   ├── cli.py           # CLI entry point
│   ├── retriever.py      # Retrieval logic
│   ├── service.py         # Experiment service
│   └── vector_store.py   # Chroma integration
├── config/
│   ├── experiment_config_local.yaml
│   └── experiment_config_docker.yaml
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