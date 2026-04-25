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
        required = ['ingestion', 'chunking', 'vector_store', 'embedding_profiles', 'generator_profiles', 'evaluation']
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

    def _resolve_workload_config(self) -> Dict[str, Any]:
        """Resolve workload limits from config."""
        default = {'max_chunks': 100, 'max_questions': 10}
        return self.config.get('workload', default)

    def run(self, output_dir: str = "./results") -> Dict[str, Any]:
        """Run the RAG evaluation experiment."""
        embedding_config = self._resolve_embedding_config()
        generator_config = self._resolve_generator_config()
        workload_config = self._resolve_workload_config()

        max_chunks = workload_config.get('max_chunks', 100)
        max_questions = workload_config.get('max_questions', 10)

        print("[INFO] Initializing RAG pipeline components...")
        print(f"  [INFO]   - Chunking strategy: {self.config['chunking'].get('strategy', 'default')}")
        print(f"  [INFO]   - Vector store: {self.config['vector_store'].get('store_type', 'chroma')}")
        print(f"  [INFO]   - Retriever k: {self.config['retriever'].get('k', 5)}")
        print(f"  [INFO]   - Workload: max_chunks={max_chunks}, max_questions={max_questions}")

        ingestion = CorpusIngestion(self.config['ingestion'])
        if 'dataset' in self.config.get('ingestion', {}):
            print(f"[INFO] Loading Hugging Face dataset: {self.config['ingestion']['dataset']}")

        chunking = ChunkingStrategy(self.config['chunking'])
        print(f"[INFO]   - Chunk size: {self.config['chunking'].get('chunk_size', 1000)}, overlap: {self.config['chunking'].get('chunk_overlap', 200)}")

        embedding = EmbeddingModel(embedding_config)
        print(f"[INFO]   - Embedding model: {embedding_config.get('model_name', 'default')}")

        vector_store_mgr = VectorStoreManager(self.config['vector_store'])
        retriever_cfg = RetrieverConfig(self.config['retriever'])
        generator = RAGGenerator(generator_config)
        print(f"[INFO]   - Generator model: {generator_config.get('model_name', 'default')}")

        evaluator = RAGEvaluator(self.config['evaluation'])

        print("[INFO] Loading documents...")
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

        print(f"[INFO] Total documents loaded: {len(documents)}")

        chunks = chunking.split_documents(documents)
        print(f"[INFO] Total chunks created: {len(chunks)}")

        chunks_to_index = chunks[:max_chunks]
        print(f"[INFO] Limiting to {len(chunks_to_index)} chunks for indexing")

        print("[INFO] Building vector store...")
        embed_model = embedding.get_embedding()
        vector_store = vector_store_mgr.get_vector_store(embed_model)
        vector_store.add_documents(chunks_to_index)
        print(f"[INFO] Indexed {len(chunks_to_index)} chunks in vector store")

        retriever = retriever_cfg.get_retriever(vector_store)
        chain = generator.get_chain(retriever)

        questions = qna_df['question'].tolist()
        reference_answers = qna_df['answer'].tolist()

        questions_to_eval = questions[:max_questions]
        reference_answers_to_eval = reference_answers[:max_questions]
        print(f"[INFO] Limiting to {len(questions_to_eval)} questions for evaluation")

        generated_answers = []
        contexts = []
        latencies = []
        costs = []

        print(f"[INFO] Running evaluation on {len(questions_to_eval)} questions...")
        for question in questions_to_eval:
            start_time = time.time()
            result = chain.invoke(question)
            latency = time.time() - start_time
            cost = 0.0

            generated_answers.append(result)
            contexts.append([])
            latencies.append(latency)
            costs.append(cost)

        print("[INFO] Evaluating results...")
        results = evaluator.evaluate_batch(questions_to_eval, reference_answers_to_eval, generated_answers, contexts, latencies, costs)

        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2)

        print(f"[INFO] Experiment completed successfully!")
        print(f"[INFO] Results saved to {output_dir}/results.json")
        print(f"[INFO] Average latency: {results.get('avg_latency', 0):.2f}s")
        print(f"[INFO] Lexical similarity: {results.get('lexical_similarity', 0):.2f}")

        return results