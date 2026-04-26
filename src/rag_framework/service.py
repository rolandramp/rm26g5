"""
RAG Experiment Service - Core Business Logic

This module contains the heart of our evaluation framework. The RAGExperimentService
class orchestrates all the different components needed to run a RAG evaluation:

1. Load documents (ingestion)
2. Split into chunks (chunking)
3. Create embeddings (embedding)
4. Store in vector database (vector_store)
5. Set up retrieval (retriever)
6. Generate answers (generator)
7. Evaluate results (evaluator)

Think of this as the "conductor" of an orchestra - it coordinates all the different
instruments (components) to play a harmonious symphony (complete evaluation).
"""

import os
import json
import time
from typing import Dict, Any, Optional

import yaml
from langchain_core.documents import Document

# Import all the components we need to coordinate
from rag_framework.ingestion import CorpusIngestion
from rag_framework.chunking import ChunkingStrategy
from rag_framework.embedding import EmbeddingModel
from rag_framework.vector_store import VectorStoreManager
from rag_framework.retriever import RetrieverConfig
from rag_framework.generator import RAGGenerator
from rag_framework.evaluation import RAGEvaluator


class RAGExperimentService:
    """
    Main service class for running RAG evaluation experiments.

    This class encapsulates all the logic needed to run a complete RAG pipeline
    evaluation. It loads configuration, initializes components, executes the
    pipeline, and collects results.

    The design follows the "service pattern" - a single class that coordinates
    multiple smaller components to accomplish a complex task.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the service with a configuration dictionary.

        Args:
            config: A dictionary containing all configuration options for the experiment.
                   This typically comes from parsing a YAML file.
        """
        self.config = config

        # Cache for loaded data - we store this to avoid reloading when running
        # multiple experiments with the same data (important for the sweep feature)
        self._cached_data = (None, None)  # (qna_df, corpus_df)

        # Validate that all required sections are present in the config
        # This fails fast with clear error messages if something is missing
        self._validate_config()

    def _validate_config(self):
        """
        Validate that the configuration contains all required sections.

        This is a "defensive programming" practice - we check for problems early
        rather than failing mysteriously later with confusing errors.
        """
        # List all the sections that must be present in any valid configuration
        required = [
            'ingestion',      # How to load documents
            'vector_store',   # Which vector database to use
            'embedding_profiles',  # Available embedding models
            'generator_profiles',  # Available language models
            'chunking_profiles',   # Available chunking strategies
            'evaluation'     # How to measure quality
        ]

        # Check which required sections are missing
        missing = [k for k in required if k not in self.config]

        # If anything is missing, raise an informative error
        if missing:
            raise ValueError(f"Missing required config sections: {missing}")

    @classmethod
    def from_config_path(cls, config_path: str) -> 'RAGExperimentService':
        """
        Factory method to create a service from a YAML configuration file.

        This is a convenient way to instantiate the service - you just provide
        the path to a config file, and this method handles reading and parsing it.

        Args:
            config_path: Path to a YAML file containing experiment configuration

        Returns:
            A new RAGExperimentService instance configured from the file
        """
        # Open the file and parse its YAML contents into a Python dictionary
        # yaml.safe_load() is the safe version that prevents executing arbitrary code
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Create and return a new instance with the loaded configuration
        return cls(config)

    def _resolve_embedding_config(self) -> Dict[str, Any]:
        """
        Determine which embedding profile to use based on configuration.

        The configuration can define multiple embedding profiles (e.g., different
        models for comparison), but only one is "active" at a time. This method
        figures out which one to use.

        Returns:
            Dictionary containing the configuration for the selected embedding model
        """
        # Look for the "active" profile name in the config
        active_profile = self.config.get('active_embedding_profile')
        # Get all available profiles (might be empty if not configured)
        embedding_profiles = self.config.get('embedding_profiles', {})

        # If an active profile is specified, find and return its config
        if active_profile:
            selected_profile = embedding_profiles.get(active_profile)

            # Validate that the named profile actually exists
            if selected_profile is None:
                available_profiles = sorted(embedding_profiles.keys())
                raise ValueError(
                    f"active_embedding_profile '{active_profile}' not found. "
                    f"Available profiles: {available_profiles}"
                )

            # Create a copy (dict()) to avoid modifying the original config
            # This is good practice - don't mutate your inputs!
            embedding_config = dict(selected_profile)
            print(f"Using embedding profile: {active_profile}")
            return embedding_config

        # No active profile specified - return empty config (use defaults)
        return {}

    def _resolve_generator_config(self) -> Dict[str, Any]:
        """
        Determine which generator (LLM) profile to use.

        Similar to embedding config resolution - we find the active generator
        from the available profiles.

        Returns:
            Dictionary containing configuration for the selected language model
        """
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
        """
        Determine which chunking profile to use.

        This method supports two modes:
        1. If profile_name is provided, look up that specific profile
        2. Otherwise, look for the active_chunking_profile in config

        This flexibility is useful when running experiments - sometimes you
        want to use the configured default, other times you specifically want
        to test a particular chunking strategy.

        Args:
            profile_name: Optional specific profile name to use (for sweeps)

        Returns:
            Dictionary containing configuration for the selected chunking strategy
        """
        # Mode 1: Specific profile requested (used in sweep mode)
        if profile_name:
            chunking_profiles = self.config.get('chunking_profiles', {})
            selected_profile = chunking_profiles.get(profile_name)
            if selected_profile is None:
                available_profiles = sorted(chunking_profiles.keys())
                raise ValueError(
                    f"chunking_profile '{profile_name}' not found. "
                    f"Available profiles: {available_profiles}"
                )
            return dict(selected_profile)

        # Mode 2: Use the configured default active profile
        active_profile = self.config.get('active_chunking_profile')
        chunking_profiles = self.config.get('chunking_profiles', {})

        if active_profile:
            selected_profile = chunking_profiles.get(active_profile)
            if selected_profile is None:
                available_profiles = sorted(chunking_profiles.keys())
                raise ValueError(
                    f"active_chunking_profile '{active_profile}' not found. "
                    f"Available profiles: {available_profiles}"
                )
            chunking_config = dict(selected_profile)
            print(f"Using chunking profile: {active_profile}")
            return chunking_config

        return {}

    def _resolve_workload_config(self) -> Dict[str, Any]:
        """
        Determine workload limits - how much data to process.

        For testing and development, we often don't want to process the entire
        dataset. This method extracts workload limits from the configuration,
        with sensible defaults if not specified.

        Returns:
            Dictionary with max_chunks and max_questions limits
        """
        # Default values ensure we always have some limits even if not configured
        # These defaults let us run quick experiments without loading thousands of chunks
        default = {'max_chunks': 100, 'max_questions': 10}
        return self.config.get('workload', default)

    def run(self, output_dir: str = "./results") -> Dict[str, Any]:
        """
        Execute a complete RAG evaluation experiment.

        This is the main method that orchestrates the entire pipeline. It follows
        a logical sequence:

        1. Resolve configurations - figure out which components to use
        2. Initialize components - create instances of all needed classes
        3. Load data - get documents and Q&A pairs
        4. Process documents - chunk and embed them
        5. Run evaluation - test each question and measure quality
        6. Save results - write metrics to disk

        Args:
            output_dir: Directory where results will be saved

        Returns:
            Dictionary containing all evaluation metrics and results
        """
        # Step 1: Resolve which profiles/configurations to use
        # Each of these looks at the config and picks the "active" one
        embedding_config = self._resolve_embedding_config()
        generator_config = self._resolve_generator_config()
        chunking_config = self._resolve_chunking_config()
        workload_config = self._resolve_workload_config()

        # Extract workload limits - how much data to process
        max_chunks = workload_config.get('max_chunks', 100)
        max_questions = workload_config.get('max_questions', 10)

        # Print startup information so user knows what's happening
        print("[INFO] Initializing RAG pipeline components...")
        print(f"  [INFO]   - Chunking strategy: {chunking_config.get('strategy', 'default')}")
        print(f"  [INFO]   - Vector store: {self.config['vector_store'].get('store_type', 'chroma')}")
        print(f"  [INFO]   - Retriever k: {self.config['retriever'].get('k', 5)}")
        print(f"  [INFO]   - Workload: max_chunks={max_chunks}, max_questions={max_questions}")

        # Step 2: Initialize all the components we'll need
        # Each component handles one part of the RAG pipeline

        # Ingestion: loads our documents and Q&A pairs from disk or HuggingFace
        ingestion = CorpusIngestion(self.config['ingestion'])
        if 'dataset' in self.config.get('ingestion', {}):
            print(f"[INFO] Loading Hugging Face dataset: {self.config['ingestion']['dataset']}")

        # Chunking: splits long documents into smaller, manageable pieces
        # The size and overlap parameters control how chunks are created
        chunking = ChunkingStrategy(chunking_config)
        print(f"[INFO]   - Chunk size: {chunking_config.get('chunk_size', 1000)}, overlap: {chunking_config.get('chunk_overlap', 200)}")

        # Embedding: converts text into numerical vectors that capture meaning
        embedding = EmbeddingModel(embedding_config)
        print(f"[INFO]   - Embedding model: {embedding_config.get('model_name', 'default')}")

        # Vector store: database that stores our embeddings for fast similarity search
        vector_store_mgr = VectorStoreManager(self.config['vector_store'])

        # Retriever: finds relevant chunks when given a question
        retriever_cfg = RetrieverConfig(self.config['retriever'])

        # Generator: the LLM that generates answers based on retrieved context
        generator = RAGGenerator(generator_config)
        print(f"[INFO]   - Generator model: {generator_config.get('model_name', 'default')}")

        # Evaluator: measures how good our results are using various metrics
        evaluator = RAGEvaluator(self.config['evaluation'], evaluator_config=embedding_config, generator_config=generator_config)

        # Step 3: Load the data we'll be working with
        print("[INFO] Loading documents...")
        qna_df, corpus_df = ingestion.load_documents()

        # Validate that we actually got data to work with
        # Without ground truth Q&A, we can't evaluate anything
        if qna_df.empty:
            raise ValueError(
                "No ground truth found. Ensure 'dataset' is configured in ingestion "
                "or provide a 'ground_truth_path' for filesystem mode."
            )

        # Convert the corpus DataFrame into LangChain Document objects
        # Each row becomes a Document with the passage text and its ID
        documents = [
            Document(page_content=str(row['passage']), metadata={'id': row['id']})
            for row in corpus_df.to_dict(orient='records')
        ]

        print(f"[INFO] Total documents loaded: {len(documents)}")

        # Step 4: Process documents - chunk them into smaller pieces
        # This is important because LLMs have context limits and we want to
        # retrieve the most relevant pieces for each question
        chunks = chunking.split_documents(documents)
        print(f"[INFO] Total chunks created: {len(chunks)}")

        # Apply workload limits - we might not want to index all chunks for quick tests
        chunks_to_index = chunks[:max_chunks]
        print(f"[INFO] Limiting to {len(chunks_to_index)} chunks for indexing")

        # Step 5: Build the vector store
        # This is where the magic happens - we embed all chunks and store them
        # so we can quickly find the most relevant ones for any question
        print("[INFO] Building vector store...")
        embed_model = embedding.get_embedding()
        vector_store = vector_store_mgr.get_vector_store(embed_model)
        vector_store.add_documents(chunks_to_index)
        print(f"[INFO] Indexed {len(chunks_to_index)} chunks in vector store")

        # Create the retriever and generator chain
        # The chain is how LangChain connects retrieval + generation
        retriever = retriever_cfg.get_retriever(vector_store)
        chain = generator.get_chain(retriever)

        # Step 6: Run the evaluation - ask each question and measure results
        # Extract questions and reference answers from our ground truth
        questions = qna_df['question'].tolist()
        reference_answers = qna_df['answer'].tolist()

        # Apply workload limits to questions too
        questions_to_eval = questions[:max_questions]
        reference_answers_to_eval = reference_answers[:max_questions]
        print(f"[INFO] Limiting to {len(questions_to_eval)} questions for evaluation")

        # Initialize result tracking lists
        generated_answers = []
        contexts = []  # Store retrieved context for each question
        latencies = []  # How long each question took
        costs = []  # Track API costs (currently always 0, placeholder for future)

        print(f"[INFO] Running evaluation on {len(questions_to_eval)} questions...")

        # Process each question one by one
        for question in questions_to_eval:
            # Time how long this takes - important metric for performance
            start_time = time.time()

            # Run the full RAG chain: retrieve relevant context + generate answer
            result = chain.invoke(question)

            # Calculate how long it took
            latency = time.time() - start_time
            cost = 0.0

            # Store the result and context for later evaluation
            generated_answers.append(result)

            # Get the context that was retrieved for this question
            # This is important for understanding why the model answered as it did
            retrieved_contexts = generator.get_last_retrieved_contexts()
            contexts.append(retrieved_contexts)
            print(f"[DEBUG] Retrieved {len(retrieved_contexts)} context chunks for question")

            # Track metrics
            latencies.append(latency)
            costs.append(cost)

        # Step 7: Evaluate all results using multiple metrics
        print("[INFO] Evaluating results...")
        results = evaluator.evaluate_batch(
            questions_to_eval,
            reference_answers_to_eval,
            generated_answers,
            contexts,
            latencies,
            costs
        )

        # Step 8: Save results to disk
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Write results as JSON for easy inspection and processing
        with open(f"{output_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2)

        # Print final summary for the user
        print(f"[INFO] Experiment completed successfully!")
        print(f"[INFO] Results saved to {output_dir}/results.json")
        print(f"[INFO] Average latency: {results.get('avg_latency', 0):.2f}s")
        print(f"[INFO] Lexical similarity: {results.get('lexical_similarity_avg', 0):.2f}")

        return results

    def run_all_combinations(self, output_dir: str = "./results") -> Dict[str, Any]:
        """
        Run experiments for ALL possible combinations of profiles.

        This is the "sweep" or "grid search" feature. Instead of running a single
        configuration, we systematically test every possible combination of:
        - All embedding models
        - All generator models
        - All chunking strategies

        This is incredibly useful for research and optimization - it lets you answer
        questions like "which embedding model works best with which chunking strategy?"

        Returns:
            Dictionary with total count and results for each combination
        """
        # Get all available profiles from config
        embedding_profiles = self.config.get('embedding_profiles', {})
        generator_profiles = self.config.get('generator_profiles', {})
        chunking_profiles = self.config.get('chunking_profiles', {})
        workload_config = self._resolve_workload_config()

        # Extract the names (keys) of each profile type
        emb_keys = list(embedding_profiles.keys())
        gen_keys = list(generator_profiles.keys())
        chunk_keys = list(chunking_profiles.keys())

        # Calculate total number of combinations (Cartesian product)
        # If we have 2 embeddings, 3 generators, and 2 chunking strategies,
        # that's 2 * 3 * 2 = 12 total experiments!
        total_combos = len(emb_keys) * len(gen_keys) * len(chunk_keys)

        print(f"[INFO] Starting sweep with {total_combos} combinations:")
        print(f"  - Embedding profiles: {emb_keys}")
        print(f"  - Generator profiles: {gen_keys}")
        print(f"  - Chunking profiles: {chunk_keys}")
        print(f"  - Workload: max_chunks={workload_config.get('max_chunks')}, max_questions={workload_config.get('max_questions')}")

        # Store results from each combination
        combos_results = []
        combo_idx = 0

        # Triple nested loop - iterate through ALL combinations
        for emb_name in emb_keys:
            emb_config = dict(embedding_profiles[emb_name])

            for gen_name in gen_keys:
                gen_config = dict(generator_profiles[gen_name])

                for chunk_name in chunk_keys:
                    combo_idx += 1
                    # Get the specific chunking config for this combination
                    chunk_config = self._resolve_chunking_config(chunk_name)

                    # Create a unique output directory for this combination
                    # Format: embed_xxx__gen_yyy__chunk_zzz
                    combo_dir = f"embed_{emb_name}__gen_{gen_name}__chunk_{chunk_name}"
                    combo_output = os.path.join(output_dir, combo_dir)

                    print(f"\n[INFO] ===== Combination {combo_idx}/{total_combos} =====")
                    print(f"[INFO] {emb_name} + {gen_name} + {chunk_name}")

                    # Run this specific combination
                    try:
                        combo_results = self._run_single_combo(
                            emb_config, gen_config, chunk_config,
                            combo_output, workload_config
                        )
                        # Tag the results with which combination produced them
                        combo_results['_combo'] = {
                            'embedding': emb_name,
                            'generator': gen_name,
                            'chunking': chunk_name
                        }
                        combos_results.append(combo_results)
                        print(f"[INFO] Combo {combo_idx} completed: latency={combo_results.get('avg_latency', 0):.2f}s, lex_sim={combo_results.get('lexical_similarity_avg', 0):.2f}")
                    except Exception as e:
                        # If this combination fails, don't stop everything!
                        # Record the error and continue with the next combination
                        print(f"[ERROR] Combo {combo_idx} failed: {e}")
                        combos_results.append({
                            '_combo': {'embedding': emb_name, 'generator': gen_name, 'chunking': chunk_name},
                            '_error': str(e)
                        })

        print(f"\n[INFO] All {total_combos} combinations completed!")

        # Save aggregated results and find best performers
        self._save_sweep_results(combos_results, output_dir, workload_config)

        return {'total_combinations': total_combos, 'results': combos_results}

    def _run_single_combo(self, emb_config: Dict, gen_config: Dict, chunk_config: Dict,
                         output_dir: str, workload_config: Dict) -> Dict[str, Any]:
        """
        Run a single experiment with a specific combination of profiles.

        This is essentially the same as the `run()` method, but it takes
        explicit configurations instead of reading from the main config.
        This allows it to be called repeatedly with different configs.

        Args:
            emb_config: Embedding model configuration
            gen_config: Generator/LLM configuration
            chunk_config: Chunking strategy configuration
            output_dir: Where to save this specific run's results
            workload_config: Limits on how much data to process

        Returns:
            Dictionary of evaluation metrics for this combination
        """
        max_chunks = workload_config.get('max_chunks', 100)
        max_questions = workload_config.get('max_questions', 10)

        # Initialize components with the specific configs for this run
        chunking = ChunkingStrategy(chunk_config)
        embedding = EmbeddingModel(emb_config)

        # Create a unique vector store for this combination
        # We use a subdirectory so each combo has its own Chroma DB
        vector_store_mgr = VectorStoreManager({
            'store_type': self.config.get('vector_store', {}).get('store_type', 'chroma'),
            'persist_directory': os.path.join(output_dir, 'chroma_db')
        })

        retriever_cfg = RetrieverConfig(self.config.get('retriever', {}))
        generator = RAGGenerator(gen_config)

        evaluator = RAGEvaluator(self.config['evaluation'], evaluator_config=emb_config, generator_config=gen_config)

        # Use cached data if available - avoids reloading for each combination
        # This is a significant optimization when running many combinations
        qna_df, corpus_df = self._cached_data
        if qna_df is None:
            ingestion = CorpusIngestion(self.config['ingestion'])
            qna_df, corpus_df = ingestion.load_documents()
            self._cached_data = (qna_df, corpus_df)

        # Convert to documents and chunk
        documents = [Document(page_content=str(row['passage']), metadata={'id': row['id']})
                     for row in corpus_df.to_dict(orient='records')]

        chunks = chunking.split_documents(documents)
        chunks_to_index = chunks[:max_chunks]

        # Build vector store
        embed_model = embedding.get_embedding()
        vector_store = vector_store_mgr.get_vector_store(embed_model)
        vector_store.add_documents(chunks_to_index)

        # Create retriever and generator chain
        retriever = retriever_cfg.get_retriever(vector_store)
        chain = generator.get_chain(retriever)

        # Get questions and reference answers
        questions = qna_df['question'].tolist()
        reference_answers = qna_df['answer'].tolist()

        questions_to_eval = questions[:max_questions]
        reference_answers_to_eval = reference_answers[:max_questions]

        # Run evaluation for each question
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

        # Evaluate results
        results = evaluator.evaluate_batch(
            questions_to_eval,
            reference_answers_to_eval,
            generated_answers,
            contexts,
            latencies,
            costs
        )

        # Save this combination's results to its own directory
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def _save_sweep_results(self, combos_results: list, output_dir: str, workload_config: Dict):
        """
        Save aggregated results from a sweep and identify best performers.

        This method creates a summary file that includes:
        1. All individual combination results
        2. The best performer for each metric

        The "best" is determined by highest value for each metric, which makes
        sense for metrics like accuracy. For metrics where lower is better
        (like latency), you might want to adjust this logic.

        Args:
            combos_results: List of result dictionaries from each combination
            output_dir: Directory to save the sweep results
            workload_config: Workload settings that were used
        """
        # Build the summary structure
        sweep_results = {
            'total_combinations': len(combos_results),
            'workload': workload_config,
            'combinations': []
        }

        # Metrics where higher is generally better
        # (You might want different logic for metrics like latency)
        numeric_metrics = ['avg_latency', 'lexical_similarity_avg', 'total_cost']

        # Process each combination's results
        for r in combos_results:
            combo = r.get('_combo', {})

            # Handle both successful runs and failed runs
            if '_error' in r:
                combo['_error'] = r['_error']
            else:
                # Extract metrics (everything except the combo identifier)
                combo['metrics'] = {k: v for k, v in r.items() if k not in ['_combo']}

            sweep_results['combinations'].append(combo)

        # Find the best performer for each metric
        if numeric_metrics:
            best_by_metric = {}
            for metric in numeric_metrics:
                best_val = None
                best_combo = None

                # Look through all results to find the best
                for r in combos_results:
                    # Skip failed runs
                    if '_error' in r:
                        continue

                    val = r.get(metric)
                    if val is not None:
                        # Update if this is better (or if it's the first one)
                        if best_val is None or val > best_val:
                            best_val = val
                            best_combo = r.get('_combo', {})

                if best_combo:
                    best_by_metric[metric] = {'combo': best_combo, 'value': best_val}

            sweep_results['best_by_metric'] = best_by_metric

        # Save to file
        with open(os.path.join(output_dir, 'sweep_results.json'), 'w') as f:
            json.dump(sweep_results, f, indent=2)

        print(f"[INFO] Sweep results saved to {output_dir}/sweep_results.json")