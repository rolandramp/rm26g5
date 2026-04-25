from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os

from rag_framework.service import RAGExperimentService

app = FastAPI(title="RAG Framework API", version="1.0.0")


class ExperimentConfig(BaseModel):
    config_path: Optional[str] = "config/experiment_config.yaml"
    output_dir: Optional[str] = "./results"
    run_all: Optional[bool] = False


class ExperimentResponse(BaseModel):
    status: str
    results: Optional[dict] = None
    message: Optional[str] = None


@app.get("/")
async def root():
    return {"service": "rag-framework", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/run-experiment", response_model=ExperimentResponse)
async def run_experiment(config: ExperimentConfig):
    try:
        service = RAGExperimentService.from_config_path(config.config_path)
        if config.run_all:
            results = service.run_all_combinations(config.output_dir)
            return ExperimentResponse(
                status="success",
                results=results,
                message=f"Sweep completed: {results.get('total_combinations')} combinations. Results saved to {config.output_dir}/sweep_results.json"
            )
        results = service.run(config.output_dir)
        return ExperimentResponse(
            status="success",
            results=results,
            message=f"Experiment completed. Results saved to {config.output_dir}/results.json"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)