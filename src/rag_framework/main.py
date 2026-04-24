"""Argparse-based CLI entrypoint for RAG evaluation framework."""

import argparse
import yaml
import json
import os
import time
from .ingestion import CorpusIngestion
from .chunking import ChunkingStrategy
from .embedding import EmbeddingModel
from .vector_store import VectorStoreManager
from .retriever import RetrieverConfig
from .generator import RAGGenerator
from .evaluation import RAGEvaluator
from langchain_core.documents import Document


def run_experiment(config_path: str, output_dir: str = "./results") -> None:
    """Run RAG evaluation experiment."""
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolve embedding profile if configured
    embedding_config = dict(config.get('embedding', {}))
    active_profile = config.get('active_embedding_profile')
    embedding_profiles = config.get('embedding_profiles', {})
    if active_profile:
        selected_profile = embedding_profiles.get(active_profile)
        if selected_profile is None:
            available_profiles = sorted(embedding_profiles.keys())
            raise ValueError(
                f"active_embedding_profile '{active_profile}' not found. "
                f"Available profiles: {available_profiles}"
            )
        embedding_config = dict(selected_profile)
        print(f"Using embedding profile: {active_profile}")

    generator_config = dict(config.get('generator', {}))
    active_gen_profile = config.get('active_generator_profile')
    generator_profiles = config.get('generator_profiles', {})
    if active_gen_profile:
        selected_gen_profile = generator_profiles.get(active_gen_profile)
        if selected_gen_profile is None:
            available_profiles = sorted(generator_profiles.keys())
            raise ValueError(
                f"active_generator_profile '{active_gen_profile}' not found. "
                f"Available profiles: {available_profiles}"
            )
        generator_config = dict(selected_gen_profile)
        print(f"Using generator profile: {active_gen_profile}")

    # Initialize components
    ingestion = CorpusIngestion(config['ingestion'])
    if 'dataset' in config.get('ingestion', {}):
        print(f"Configured to load Hugging Face dataset: {config['ingestion']['dataset']}")
    chunking = ChunkingStrategy(config['chunking'])
    embedding = EmbeddingModel(embedding_config)
    vector_store_mgr = VectorStoreManager(config['vector_store'])
    retriever_cfg = RetrieverConfig(config['retriever'])
    generator = RAGGenerator(generator_config)
    evaluator = RAGEvaluator(config['evaluation'])

    # Load and process documents
    qna_df, corpus_df = ingestion.load_documents()

    if qna_df.empty:
        raise ValueError(
            "No ground truth found. Ensure 'dataset' is configured in ingestion "
            "or provide a 'ground_truth_path' for filesystem mode."
        )

    # Build Document objects from corpus DataFrame for chunking
    documents = [
        Document(page_content=str(row['passage']), metadata={'id': row['id']})
        for row in corpus_df.to_dict(orient='records')
    ]

    chunks = chunking.split_documents(documents)

    # Set up vector store
    embed_model = embedding.get_embedding()
    vector_store = vector_store_mgr.get_vector_store(embed_model)
    vector_store.add_documents(chunks[1:100]) # only indexing a subset for demo purposes - adjust as needed

    # Set up retriever and generator
    retriever = retriever_cfg.get_retriever(vector_store)
    chain = generator.get_chain(retriever)

    # Run evaluation using ground truth from dataset
    questions = qna_df['question'].tolist()
    reference_answers = qna_df['answer'].tolist()

    generated_answers = []
    contexts = []
    latencies = []
    costs = []

    for question in questions[1:10]:  # limit to first 10 for demo - adjust as needed
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
