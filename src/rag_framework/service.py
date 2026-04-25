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
        self._cached_data = (None, None)  # (qna_df, corpus_df)
        self._validate_config()

    def _validate_config(self):
        """Validate required config sections."""
        required = ['ingestion', 'chunking', 'vector_store', 'embedding_profiles', 'generator_profiles', 'chunking_profiles', 'evaluation']
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

        return {}

    def _resolve_generator_config(self) -> Dict[str, Any]:
        """Resolve active generator profile."""
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

        return {}

    def _resolve_chunking_config(self, profile_name: str = None) -> Dict[str, Any]:
        """Resolve chunking profile."""
        chunking_profiles = self.config.get('chunking_profiles', {})

        if profile_name:
            selected_profile = chunking_profiles.get(profile_name)
            if selected_profile is None:
                available = sorted(chunking_profiles.keys())
                raise ValueError(f"chunking_profile '{profile_name}' not found. Available: {available}")
            return {
                'strategy': self.config.get('chunking', {}).get('strategy', 'recursive'),
                **selected_profile
            }

        return dict(self.config.get('chunking', {}))

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

        evaluator = RAGEvaluator(self.config['evaluation'], evaluator_config=embedding_config, generator_config=generator_config)

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
            retrieved_contexts = generator.get_last_retrieved_contexts()
            contexts.append(retrieved_contexts)
            print(f"[DEBUG] Retrieved {len(retrieved_contexts)} context chunks for question")
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
        print(f"[INFO] Lexical similarity: {results.get('lexical_similarity_avg', 0):.2f}")

        return results

    def run_all_combinations(self, output_dir: str = "./results") -> Dict[str, Any]:
        """Run all combinations of embedding x generator x chunking profiles."""
        embedding_profiles = self.config.get('embedding_profiles', {})
        generator_profiles = self.config.get('generator_profiles', {})
        chunking_profiles = self.config.get('chunking_profiles', {})
        workload_config = self._resolve_workload_config()

        emb_keys = list(embedding_profiles.keys())
        gen_keys = list(generator_profiles.keys())
        chunk_keys = list(chunking_profiles.keys())

        total_combos = len(emb_keys) * len(gen_keys) * len(chunk_keys)
        print(f"[INFO] Starting sweep with {total_combos} combinations:")
        print(f"  - Embedding profiles: {emb_keys}")
        print(f"  - Generator profiles: {gen_keys}")
        print(f"  - Chunking profiles: {chunk_keys}")
        print(f"  - Workload: max_chunks={workload_config.get('max_chunks')}, max_questions={workload_config.get('max_questions')}")

        combos_results = []
        combo_idx = 0

        for emb_name in emb_keys:
            emb_config = dict(embedding_profiles[emb_name])

            for gen_name in gen_keys:
                gen_config = dict(generator_profiles[gen_name])

                for chunk_name in chunk_keys:
                    combo_idx += 1
                    chunk_config = self._resolve_chunking_config(chunk_name)

                    combo_dir = f"embed_{emb_name}__gen_{gen_name}__chunk_{chunk_name}"
                    combo_output = os.path.join(output_dir, combo_dir)

                    print(f"\n[INFO] ===== Combination {combo_idx}/{total_combos} =====")
                    print(f"[INFO] {emb_name} + {gen_name} + {chunk_name}")

                    try:
                        combo_results = self._run_single_combo(
                            emb_config, gen_config, chunk_config,
                            combo_output, workload_config
                        )
                        combo_results['_combo'] = {
                            'embedding': emb_name,
                            'generator': gen_name,
                            'chunking': chunk_name
                        }
                        combos_results.append(combo_results)
                        print(f"[INFO] Combo {combo_idx} completed: latency={combo_results.get('avg_latency', 0):.2f}s, lex_sim={combo_results.get('lexical_similarity_avg', 0):.2f}")
                    except Exception as e:
                        print(f"[ERROR] Combo {combo_idx} failed: {e}")
                        combos_results.append({
                            '_combo': {'embedding': emb_name, 'generator': gen_name, 'chunking': chunk_name},
                            '_error': str(e)
                        })

        print(f"\n[INFO] All {total_combos} combinations completed!")
        self._save_sweep_results(combos_results, output_dir, workload_config)

        return {'total_combinations': total_combos, 'results': combos_results}

    def _run_single_combo(self, emb_config: Dict, gen_config: Dict, chunk_config: Dict,
                         output_dir: str, workload_config: Dict) -> Dict[str, Any]:
        """Run a single combination of profiles."""
        max_chunks = workload_config.get('max_chunks', 100)
        max_questions = workload_config.get('max_questions', 10)

        chunking = ChunkingStrategy(chunk_config)
        embedding = EmbeddingModel(emb_config)
        vector_store_mgr = VectorStoreManager({
            'store_type': self.config.get('vector_store', {}).get('store_type', 'chroma'),
            'persist_directory': os.path.join(output_dir, 'chroma_db')
        })
        retriever_cfg = RetrieverConfig(self.config.get('retriever', {}))
        generator = RAGGenerator(gen_config)

        evaluator = RAGEvaluator(self.config['evaluation'], evaluator_config=emb_config, generator_config=gen_config)

        qna_df, corpus_df = self._cached_data
        if qna_df is None:
            ingestion = CorpusIngestion(self.config['ingestion'])
            qna_df, corpus_df = ingestion.load_documents()
            self._cached_data = (qna_df, corpus_df)

        documents = [Document(page_content=str(row['passage']), metadata={'id': row['id']})
                     for row in corpus_df.to_dict(orient='records')]

        chunks = chunking.split_documents(documents)
        chunks_to_index = chunks[:max_chunks]

        embed_model = embedding.get_embedding()
        vector_store = vector_store_mgr.get_vector_store(embed_model)
        vector_store.add_documents(chunks_to_index)

        retriever = retriever_cfg.get_retriever(vector_store)
        chain = generator.get_chain(retriever)

        questions = qna_df['question'].tolist()
        reference_answers = qna_df['answer'].tolist()

        questions_to_eval = questions[:max_questions]
        reference_answers_to_eval = reference_answers[:max_questions]

        generated_answers = []
        contexts = []
        latencies = []
        costs = []

        for question in questions_to_eval:
            start_time = time.time()
            result = chain.invoke(question)
            latency = time.time() - start_time

            generated_answers.append(result)
            retrieved_contexts = generator.get_last_retrieved_contexts()
            contexts.append(retrieved_contexts)
            latencies.append(latency)
            costs.append(0.0)

        results = evaluator.evaluate_batch(questions_to_eval, reference_answers_to_eval, generated_answers, contexts, latencies, costs)

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def _save_sweep_results(self, combos_results: list, output_dir: str, workload_config: Dict):
        """Save aggregated sweep results."""
        sweep_results = {
            'total_combinations': len(combos_results),
            'workload': workload_config,
            'combinations': []
        }

        numeric_metrics = ['avg_latency', 'lexical_similarity_avg', 'total_cost']

        for r in combos_results:
            combo = r.get('_combo', {})
            if '_error' in r:
                combo['_error'] = r['_error']
            else:
                combo['metrics'] = {k: v for k, v in r.items() if k not in ['_combo']}
            sweep_results['combinations'].append(combo)

        if numeric_metrics:
            best_by_metric = {}
            for metric in numeric_metrics:
                best_val = None
                best_combo = None
                for r in combos_results:
                    if '_error' in r:
                        continue
                    val = r.get(metric)
                    if val is not None:
                        if best_val is None or val > best_val:
                            best_val = val
                            best_combo = r.get('_combo', {})
                if best_combo:
                    best_by_metric[metric] = {'combo': best_combo, 'value': best_val}
            sweep_results['best_by_metric'] = best_by_metric

        with open(os.path.join(output_dir, 'sweep_results.json'), 'w') as f:
            json.dump(sweep_results, f, indent=2)

        print(f"[INFO] Sweep results saved to {output_dir}/sweep_results.json")