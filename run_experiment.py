import sys
import argparse
from pathlib import Path

# Ensure local `src/` package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag_framework.main import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run RAG experiment without installing extras")
    parser.add_argument("config", nargs="?", default="config/experiment_config.yaml", help="Path to experiment YAML config")
    parser.add_argument("--output", "-o", dest="output", default="results", help="Output directory for results")
    args = parser.parse_args()

    run_experiment(args.config, args.output)


if __name__ == "__main__":
    main()
