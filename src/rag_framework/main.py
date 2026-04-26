"""
RAG Evaluation Framework - Command Line Interface

This module provides the main entry point for running RAG (Retrieval-Augmented Generation)
evaluation experiments from the command line. It uses argparse to parse user arguments
and delegates the actual experiment execution to the RAGExperimentService.

Think of this as the "front door" of our application - it's what users interact with
when they run the tool from their terminal.
"""

import argparse
from rag_framework.service import RAGExperimentService


def run_experiment(config_path: str, output_dir: str = "./results", run_all: bool = False) -> dict:
    """
    Execute a RAG evaluation experiment.

    This function serves as a bridge between the CLI and the core business logic.
    It creates a RAGExperimentService instance from a configuration file and runs
    either a single experiment or all possible combinations of components.

    Args:
        config_path: Path to the YAML configuration file that defines the experiment
        output_dir: Directory where results will be saved (default: "./results")
        run_all: If True, runs all combinations of embedding/generator/chunking profiles

    Returns:
        A dictionary containing experiment results and metrics
    """
    # Load the configuration and create the service instance
    # The service handles all the heavy lifting of running experiments
    service = RAGExperimentService.from_config_path(config_path)

    # Decide whether to run a single experiment or sweep through all combinations
    # "run_all" is useful for comparing different configurations systematically
    if run_all:
        return service.run_all_combinations(output_dir)
    return service.run(output_dir)


def main(argv=None):
    """
    Main entry point for the CLI application.

    This sets up the argument parser with subcommands. We use subparsers to allow
    future extension (e.g., adding 'serve', 'evaluate' commands) without breaking
    existing usage patterns.
    """
    # Create the argument parser with a descriptive help message
    parser = argparse.ArgumentParser(
        prog="rag-eval",
        description="Parametrizable RAG Evaluation Framework CLI"
    )

    # Add subparsers to handle different commands
    # The 'dest' parameter stores which subcommand was chosen in args.command
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Define the 'run-experiment' subcommand for executing evaluations
    run_parser = subparsers.add_parser(
        'run-experiment',
        help='Run RAG evaluation experiment'
    )

    # Required: path to the YAML configuration file
    # This is the "blueprint" that tells the framework what components to use
    run_parser.add_argument(
        '--config',
        dest='config_path',
        required=True,
        help='Path to experiment configuration YAML file'
    )

    # Optional: where to save results (defaults to ./results)
    run_parser.add_argument(
        '--output',
        dest='output_dir',
        default='./results',
        help='Directory to save results'
    )

    # Optional flag: when present, runs ALL combinations of component profiles
    # This enables systematic comparison across different embedding/generator/chunking options
    run_parser.add_argument(
        '--run-all',
        action='store_true',
        help='Run all combinations of embedding x generator x chunking profiles'
    )

    # Parse the command-line arguments
    args = parser.parse_args(argv)

    # Execute the chosen command
    if args.command == 'run-experiment':
        # Run the experiment and capture results
        results = run_experiment(args.config_path, args.output_dir, args.run_all)

        # Display user-friendly summary based on what was run
        if args.run_all:
            # For sweeps, show total combinations and output location
            print(f"Sweep completed: {results.get('total_combinations')} combinations processed")
            print(f"Results saved to: {args.output_dir}/sweep_results.json")
        else:
            # For single runs, show key performance metrics
            print(f"Experiment completed: latency={results.get('avg_latency', 0):.2f}s, lex_sim={results.get('lexical_similarity_avg', 0):.2f}")


if __name__ == '__main__':
    # Standard Python idiom: only run main() when script is executed directly,
    # not when imported as a module
    main()