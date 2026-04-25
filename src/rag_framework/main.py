"""Argparse-based CLI entrypoint for RAG evaluation framework."""

import argparse

from rag_framework.service import RAGExperimentService


def run_experiment(config_path: str, output_dir: str = "./results") -> dict:
    """Run RAG evaluation experiment."""
    service = RAGExperimentService.from_config_path(config_path)
    return service.run(output_dir)


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