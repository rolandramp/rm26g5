"""Argparse-based CLI entrypoint for RAG evaluation framework."""

import argparse

from rag_framework.service import RAGExperimentService


def run_experiment(config_path: str, output_dir: str = "./results", run_all: bool = False) -> dict:
    """Run RAG evaluation experiment."""
    service = RAGExperimentService.from_config_path(config_path)
    if run_all:
        return service.run_all_combinations(output_dir)
    return service.run(output_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rag-eval", description="Parametrizable RAG Evaluation Framework CLI")
    subparsers = parser.add_subparsers(dest='command', required=True)

    run_parser = subparsers.add_parser('run-experiment', help='Run RAG evaluation experiment')
    run_parser.add_argument('--config', dest='config_path', required=True, help='Path to experiment configuration YAML file')
    run_parser.add_argument('--output', dest='output_dir', default='./results', help='Directory to save results')
    run_parser.add_argument('--run-all', action='store_true', help='Run all combinations of embedding x generator x chunking profiles')

    args = parser.parse_args(argv)

    if args.command == 'run-experiment':
        results = run_experiment(args.config_path, args.output_dir, args.run_all)
        if args.run_all:
            print(f"Sweep completed: {results.get('total_combinations')} combinations processed")
            print(f"Results saved to: {args.output_dir}/sweep_results.json")
        else:
            print(f"Experiment completed: latency={results.get('avg_latency', 0):.2f}s, lex_sim={results.get('lexical_similarity_avg', 0):.2f}")


if __name__ == '__main__':
    main()