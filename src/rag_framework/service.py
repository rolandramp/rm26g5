"""RAG Experiment Service - Core business logic for running RAG evaluation experiments."""

import os
import json
import time
from typing import Dict, Any, Optional

import yaml
from langchain_core.documents import Document

from rag_framework.ingestion import CorpusIngestion
from rag_framework.chunking import ChunkingStrategy
from rag_framework.embedding import EmbeddingModel
from rag_framework.vector_store import VectorStoreManager
from rag_framework.retriever import RetrieverConfig
from rag_framework.generator import RAGGenerator
from rag_framework.evaluation import RAGEvaluator


class RAGExperimentService:
    """Service class for running RAG evaluation experiments."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._validate_config()

    def _validate_config(self):
        """Validate required config sections."""
        required = ['ingestion', 'chunking', 'vector_store', 'retriever', 'generator', 'evaluation']
        missing = [k for k in required if k not in self.config]
        if missing:
            raise ValueError(f"Missing required config sections: {missing}")

    @classmethod
    def from_config_path(cls, config_path: str) -> 'RAGExperimentService':
        """Create service from YAML config file."""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls(config)

    def _resolve_embedding_config(self) -> Dict[str, Any]:
        """Resolve active embedding profile."""
        embedding_config = dict(self.config.get('embedding', {}))
        active_profile = self.config.get('active_embedding_profile')
        embedding_profiles = self.config.get('embedding_profiles', {})

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

        return embedding_config

    def _resolve_generator_config(self) -> Dict[str, Any]:
        """Resolve active generator profile."""
        generator_config = dict(self.config.get('generator', {}))
        active_gen_profile = self.config.get('active_generator_profile')
        generator_profiles = self.config.get('generator_profiles', {})

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

        return generator_config

    def run(self, output_dir: str = "./results") -> Dict[str, Any]:
        """Run the RAG evaluation experiment."""
        embedding_config = self._resolve_embedding_config()
        generator_config = self._resolve_generator_config()

        ingestion = CorpusIngestion(self.config['ingestion'])
        if 'dataset' in self.config.get('ingestion', {}):
            print(f"Configured to load Hugging Face dataset: {self.config['ingestion']['dataset']}")

        chunking = ChunkingStrategy(self.config['chunking'])
        embedding = EmbeddingModel(embedding_config)
        vector_store_mgr = VectorStoreManager(self.config['vector_store'])
        retriever_cfg = RetrieverConfig(self.config['retriever'])
        generator = RAGGenerator(generator_config)
        evaluator = RAGEvaluator(self.config['evaluation'])

        qna_df, corpus_df = ingestion.load_documents()

        if qna_df.empty:
            raise ValueError(
                "No ground truth found. Ensure 'dataset' is configured in ingestion "
                "or provide a 'ground_truth_path' for filesystem mode."
            )

        documents = [
            Document(page_content=str(row['passage']), metadata={'id': row['id']})
            for row in corpus_df.to_dict(orient='records')
        ]

        chunks = chunking.split_documents(documents)

        embed_model = embedding.get_embedding()
        vector_store = vector_store_mgr.get_vector_store(embed_model)
        vector_store.add_documents(chunks[1:100])

        retriever = retriever_cfg.get_retriever(vector_store)
        chain = generator.get_chain(retriever)

        questions = qna_df['question'].tolist()
        reference_answers = qna_df['answer'].tolist()

        generated_answers = []
        contexts = []
        latencies = []
        costs = []

        for question in questions[1:10]:
            start_time = time.time()
            result = chain.invoke(question)
            latency = time.time() - start_time
            cost = 0.0

            generated_answers.append(result)
            contexts.append([])
            latencies.append(latency)
            costs.append(cost)

        results = evaluator.evaluate_batch(questions, reference_answers, generated_answers, contexts, latencies, costs)

        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Experiment completed. Results saved to {output_dir}/results.json")

        return results