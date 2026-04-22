# rm26g5
Research Methods 2026 Group 5

## Parametrizable RAG Evaluation Framework

A Python 3.14 project for evaluating Retrieval-Augmented Generation (RAG) pipelines against ground-truth datasets.

### Features

- Configurable document loaders, chunking strategies, embedding models, vector databases, retrievers, and LLMs
- Support for OpenAI-compatible REST APIs
- Local persistent vector storage with Chroma
- Comprehensive evaluation metrics (semantic similarity, lexical similarity, latency, cost, retrieval quality)
- CLI for parameterized experiments
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

### Usage

1. Prepare your data:
   - Place documents in `./data/`
   - Create ground truth JSON at `./data/ground_truth.json` with format:
     ```json
     [
       {"question": "What is RAG?", "answer": "Retrieval-Augmented Generation..."},
       ...
     ]
     ```

2. Configure experiment in `config/experiment_config.yaml`

3. Run experiment:
```bash
rag-eval run-experiment --config config/experiment_config.yaml --output results/
```

### Project Structure

- `src/rag_framework/` - Core modules
- `config/` - Configuration files
- `data/` - Input data and ground truth
- `results/` - Experiment outputs
- `tests/` - Unit tests
