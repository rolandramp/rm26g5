"""Argparse-based CLI entrypoint for RAG evaluation framework."""

import argparse
import yaml
import json
import os
import time
from pathlib import Path
from .ingestion import CorpusIngestion
from .chunking import ChunkingStrategy
from .embedding import EmbeddingModel
from .vector_store import VectorStoreManager
from .retriever import RetrieverConfig
from .generator import RAGGenerator
from .evaluation import RAGEvaluator


def run_experiment(config_path: str, output_dir: str = "./results") -> None:
    """Run RAG evaluation experiment."""
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize components
    ingestion = CorpusIngestion(config['ingestion'])
    chunking = ChunkingStrategy(config['chunking'])
    embedding = EmbeddingModel(config['embedding'])
    vector_store_mgr = VectorStoreManager(config['vector_store'])
    retriever_cfg = RetrieverConfig(config['retriever'])
    generator = RAGGenerator(config['generator'])
    evaluator = RAGEvaluator(config['evaluation'])

    # Load and process documents
    documents = ingestion.load_documents()
    chunks = chunking.split_documents(documents)

    # Set up vector store
    embed_model = embedding.get_embedding()
    vector_store = vector_store_mgr.get_vector_store(embed_model)
    vector_store.add_documents(chunks)

    # Set up retriever and generator
    retriever = retriever_cfg.get_retriever(vector_store)
    chain = generator.get_chain(retriever)

    # Load ground truth
    with open(config['ground_truth_path']) as f:
        ground_truth = json.load(f)

    # Run evaluation
    questions = [item['question'] for item in ground_truth]
    reference_answers = [item['answer'] for item in ground_truth]

    generated_answers = []
    contexts = []
    latencies = []
    costs = []

    for question in questions:
        start_time = time.time()
        result = chain.invoke(question)
        latency = time.time() - start_time
        # Placeholder for cost calculation
        cost = 0.0  # Implement based on token usage

        generated_answers.append(result)
        # Placeholder for contexts - need to extract from chain
        contexts.append([])  # Implement context extraction
        latencies.append(latency)
        costs.append(cost)

    # Evaluate
    results = evaluator.evaluate_batch(questions, reference_answers, generated_answers, contexts, latencies, costs)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Experiment completed. Results saved to {output_dir}/results.json")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rag-eval", description="Parametrizable RAG Evaluation Framework CLI")
    subparsers = parser.add_subparsers(dest='command', required=True)

    run_parser = subparsers.add_parser('run-experiment', help='Run RAG evaluation experiment')
    run_parser.add_argument('--config', dest='config_path', required=True, help='Path to experiment configuration YAML file')
    run_parser.add_argument('--output', dest='output_dir', default='./results', help='Directory to save results')

    args = parser.parse_args(argv)

    if args.command == 'run-experiment':
        run_experiment(args.config_path, args.output_dir)


if __name__ == '__main__':
    main()
