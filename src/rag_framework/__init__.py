"""
RAG Evaluation Framework

A comprehensive framework for evaluating Retrieval-Augmented Generation (RAG) systems.

WHAT IS RAG?
Retrieval-Augmented Generation is a technique where a language model can access
external information (a knowledge base) when generating responses. Instead of only
using knowledge baked into the model, it retrieves relevant documents to inform its answer.

This framework helps you measure how well your RAG system is working by:
1. Running your pipeline on test questions
2. Comparing generated answers to known correct answers
3. Computing various quality metrics

COMPONENTS:
- main.py: Command-line interface for running experiments
- service.py: Core orchestration logic
- ingestion.py: Loading documents and Q&A data
- chunking.py: Splitting documents into searchable chunks
- embedding.py: Converting text to vector embeddings
- vector_store.py: Storing embeddings for fast search
- retriever.py: Finding relevant documents for questions
- generator.py: Using LLM to generate answers from context
- evaluation.py: Computing quality metrics
- api.py: REST API for programmatic access

GETTING STARTED:
1. Create a YAML config file (see config/ directory for examples)
2. Run: python -m rag_framework.main run-experiment --config your_config.yaml

For API mode:
1. Run: python -m rag_framework.api
2. Send POST request to http://localhost:8000/run-experiment
"""

__version__ = "0.1.0"