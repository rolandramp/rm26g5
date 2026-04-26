"""
RAG Framework REST API

This module provides a web-based API for running RAG evaluations.
Instead of using the command-line interface, you can make HTTP requests.

WHY USE AN API?
- Easier integration with other systems
- Can be called from web applications
- Allows remote execution (run experiments on a server)
- Better for certain orchestration workflows

This uses FastAPI, a modern Python web framework that's fast and easy to use.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os

from rag_framework.service import RAGExperimentService

# Create the FastAPI application
# FastAPI automatically generates API documentation at /docs
app = FastAPI(title="RAG Framework API", version="1.0.0")


class ExperimentConfig(BaseModel):
    """
    Request model for experiment configuration.

    Pydantic automatically validates the incoming JSON and ensures
    types are correct. This makes the API robust against bad input.
    """
    config_path: Optional[str] = os.environ.get("CONFIG_PATH", "config/experiment_config_local.yaml")
    output_dir: Optional[str] = "./results"
    run_all: Optional[bool] = False


class ExperimentResponse(BaseModel):
    """
    Response model for experiment results.

    All API responses follow this structure for consistency.
    """
    status: str
    results: Optional[dict] = None
    message: Optional[str] = None


@app.get("/")
async def root():
    """
    Root endpoint - basic info about the service.

    Returns a simple welcome message with version info.
    """
    return {"service": "rag-framework", "version": "1.0.0"}


@app.get("/health")
async def health():
    """
    Health check endpoint.

    Kubernetes and other orchestration systems use this to check
    if the service is running and healthy.
    """
    return {"status": "healthy"}


@app.post("/run-experiment", response_model=ExperimentResponse)
async def run_experiment(config: ExperimentConfig):
    """
    Run a RAG evaluation experiment via API.

    This is the main endpoint - send a configuration and get back
    evaluation results. Can run either a single experiment or sweep
    through all combinations.

    The async/await pattern allows handling multiple requests concurrently,
    which is important when running long experiments.

    Args:
        config: ExperimentConfig with path to config and options

    Returns:
        ExperimentResponse with status and results
    """
    try:
        # Load the experiment configuration
        service = RAGExperimentService.from_config_path(config.config_path)

        if config.run_all:
            # Run all combinations (sweep)
            results = service.run_all_combinations(config.output_dir)
            return ExperimentResponse(
                status="success",
                results=results,
                message=f"Sweep completed: {results.get('total_combinations')} combinations. Results saved to {config.output_dir}/sweep_results.json"
            )
        else:
            # Run single experiment
            results = service.run(config.output_dir)
            return ExperimentResponse(
                status="success",
                results=results,
                message=f"Experiment completed. Results saved to {config.output_dir}/results.json"
            )
    except Exception as e:
        # If anything goes wrong, return a 500 error with details
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # Default port is 8000, but can be overridden via environment variable
    # This allows running in containerized environments easily
    port = int(os.environ.get("PORT", 8000))

    # Run the FastAPI app with uvicorn server
    # host="0.0.0.0" makes it accessible from outside the container/machine
    uvicorn.run(app, host="0.0.0.0", port=port)