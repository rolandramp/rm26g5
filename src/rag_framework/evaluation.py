"""
Evaluation Module

This module measures how well the RAG pipeline performs. Evaluation is
crucial for understanding if your system is working and for comparing
different configurations.

WHY IS EVALUATION IMPORTANT?
A RAG system can fail in many ways:
- Retrieval might return irrelevant documents
- Generation might hallucinate (make up facts)
- The combination might produce wrong answers

Without measurement, you can't improve! We use multiple metrics to get
a complete picture of system quality.

METRICS WE USE:
1. RAGAS metrics (from the RAGAS library):
   - Semantic Similarity: How similar is the answer to the reference?
   - Answer Correctness: Did we get the right answer?
   - Context Precision: Did we retrieve the right context?
   - Bleu/Rouge Scores: Classical NLP metrics for text similarity
   - Faithfulness: Does the answer stick to the provided context?

2. Custom metrics:
   - Lexical Similarity: Simple word overlap between generated and reference
   - Latency: How fast was each question answered
   - Retrieval Quality: Stats about what was retrieved
"""

from typing import Dict, Any, List, Optional
from ragas import evaluate
from ragas.metrics import (
    SemanticSimilarity,
    AnswerCorrectness,
    ContextPrecision,
    BleuScore,
    RougeScore,
    Faithfulness
)
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rag_framework.embedding import OllamaCompatibleEmbeddings
from datasets import Dataset
import numpy as np


class RAGEvaluator:
    """
    Evaluates RAG pipeline performance using multiple metrics.

    This class runs comprehensive evaluation on the RAG pipeline's outputs.
    It combines RAGAS (a specialized RAG evaluation framework) with custom
    metrics for a complete picture.
    """

    def __init__(self, config: Dict[str, Any], evaluator_config: Optional[Dict[str, Any]] = None, generator_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the evaluator.

        Args:
            config: Evaluation configuration
            evaluator_config: Configuration for the embedding model used in evaluation
            generator_config: Configuration for the LLM used in evaluation
        """
        self.config = config
        self.evaluator_config = evaluator_config or {}
        self.generator_config = generator_config or {}

    def _get_evaluator_llm(self):
        """
        Get the language model used for evaluation.

        RAGAS uses an LLM to evaluate aspects like "faithfulness" and
        "answer correctness" - it basically asks the LLM to judge quality.

        Returns:
            A ChatOpenAI instance (works with Ollama too)
        """
        model_name = self.generator_config.get('model_name', 'gpt-3.5-turbo')
        api_key = self.generator_config.get('api_key', 'ollama')
        base_url = self.generator_config.get('base_url')

        print(f"[DEBUG] Creating evaluator LLM: model={model_name}, base_url={base_url}")

        # Temperature 0 for deterministic evaluation results
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0
        )

    def _get_evaluator_embeddings(self):
        """
        Get the embedding model used for evaluation.

        RAGAS needs embeddings to compute semantic similarity.

        Returns:
            A LangchainEmbeddingsWrapper around our Ollama-compatible embeddings
        """
        model_name = self.evaluator_config.get('model_name', 'nomic-embed-text')
        api_key = self.evaluator_config.get('api_key', 'ollama')
        base_url = self.evaluator_config.get('base_url')

        print(f"[DEBUG] Creating evaluator embeddings: model={model_name}, base_url={base_url}")

        return LangchainEmbeddingsWrapper(
            OllamaCompatibleEmbeddings(
                model=model_name,
                api_key=api_key,
                base_url=base_url
            )
        )

    def evaluate_batch(self, questions, reference_answers,
                    generated_answers, contexts,
                    latencies):
        """
        Compute all metrics for a batch of evaluation results.

        This is the main entry point - call this with your evaluation
        data and get back a dictionary of metrics.

        Args:
            questions: List of questions that were asked
            reference_answers: The known correct answers (ground truth)
            generated_answers: What our RAG system produced
            contexts: What context was retrieved for each question
            latencies: How long each question took to answer

        Returns:
            Dictionary with all computed metrics
        """
        results = {}

        # Format contexts consistently - they might be strings or Document objects
        formatted_contexts = []
        for ctx_list in contexts:
            if not ctx_list:
                # Empty context
                formatted_contexts.append([])
            elif isinstance(ctx_list[0], str):
                # Already strings
                formatted_contexts.append(ctx_list)
            else:
                # Document objects - keep as is (LangchainEmbeddingsWrapper handles them)
                formatted_contexts.append(ctx_list)

        # Create a RAGAS dataset from our results
        # RAGAS expects specific column names
        data = {
            "question": questions,
            "answer": generated_answers,
            "contexts": formatted_contexts,
            "ground_truth": reference_answers
        }
        dataset = Dataset.from_dict(data)

        # Try running RAGAS evaluation (might fail if API is unavailable)
        try:
            print("[DEBUG] Setting up evaluator LLM for RAGAS...")
            evaluator_llm = self._get_evaluator_llm()
            evaluator_embeddings = self._get_evaluator_embeddings()

            print("[DEBUG] Running RAGAS evaluation with metrics...")
            # Define which RAGAS metrics to compute
            # Each measures a different aspect of quality
            ragas_metrics = [
                SemanticSimilarity(),     # Semantic similarity to ground truth
                AnswerCorrectness(),      # Did we answer correctly?
                ContextPrecision(),       # Did we retrieve the right context?
                BleuScore(),              # N-gram overlap (classical)
                RougeScore(),             # Recall-oriented overlap
                Faithfulness()            # Does answer follow from context?
            ]

            # Run the evaluation
            ragas_results = evaluate(
                dataset,
                metrics=ragas_metrics,
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
                batch_size=8  # Process in batches for efficiency
            )

            # Extract individual metrics from the results
            results['context_precision_avg'] = np.nanmean(ragas_results['context_precision'])
            results['faithfulness_avg'] = np.nanmean(ragas_results['faithfulness'])
            results['semantic_similarity_avg'] = np.nanmean(ragas_results['semantic_similarity'])
            results['answer_correctness_avg'] = np.nanmean(ragas_results['answer_correctness'])
            results['bleu_score_avg'] = np.nanmean(ragas_results['bleu_score'])
            results['rouge_score_avg'] = np.nanmean(ragas_results['rouge_score(mode=fmeasure)'])

        except Exception as e:
            # If RAGAS fails (e.g., API unavailable), fall back to basic metrics
            # Don't let evaluation failure crash the whole experiment
            print(f"[WARNING] RAGAS evaluation failed: {e}")
            print("[DEBUG] Falling back to basic metrics only")

        # Compute custom metrics that don't require external APIs
        lexical_similarity_per_question = self._compute_lexical_similarity(generated_answers, reference_answers)
        results['lexical_similarity_avg'] = float(np.mean(lexical_similarity_per_question)) if lexical_similarity_per_question else 0.0
        results['avg_latency'] = float(np.mean(latencies)) if latencies else 0.0
        results['retrieval_quality'] = self._compute_retrieval_quality(formatted_contexts, reference_answers)

        return results

    def _compute_lexical_similarity(self, generated, reference):
        """
        Compute simple word overlap similarity.

        This is a basic metric that measures how many words overlap
        between the generated and reference answers. It's not perfect
        (phrase order matters), but it's fast and requires no API calls.

        Uses Jaccard similarity: intersection / union of word sets.

        Args:
            generated: List of generated answer strings
            reference: List of reference answer strings

        Returns:
            List of similarity scores (0 to 1) for each question
        """
        scores = []
        for gen, ref in zip(generated, reference):
            # Convert to lowercase and split into words
            gen_words = set(gen.lower().split()) if gen else set()
            ref_words = set(ref.lower().split()) if ref else set()

            if gen_words and ref_words:
                # Jaccard similarity: intersection / union
                overlap = len(gen_words & ref_words) / len(gen_words | ref_words)
                scores.append(overlap)
            else:
                # Handle edge cases (empty strings, etc.)
                scores.append(0.0)

        return scores

    def _compute_retrieval_quality(self, contexts, reference_answers):
        """
        Compute metrics about retrieval performance.

        This tells us about the retrieval side of RAG:
        - How many chunks were retrieved per question?
        - How long were those chunks?
        - Did we successfully retrieve anything?

        These stats help diagnose retrieval problems.

        Args:
            contexts: List of retrieved context lists
            reference_answers: Reference answers (unused but kept for signature)

        Returns:
            Dictionary with retrieval statistics
        """
        # Handle empty or all-empty case
        if not contexts or all(not c for c in contexts):
            return {
                'avg_contexts_retrieved': 0.0,
                'avg_context_length': 0.0,
                'questions_with_contexts': 0
            }

        total_contexts = 0
        total_length = 0
        questions_with_contexts = 0

        for ctx_list in contexts:
            if ctx_list:
                questions_with_contexts += 1
                total_contexts += len(ctx_list)
                # Sum up the length of each context chunk
                for ctx in ctx_list:
                    if isinstance(ctx, str):
                        total_length += len(ctx)
                    elif hasattr(ctx, 'page_content'):
                        total_length += len(ctx.page_content)

        num_questions = len(contexts) if contexts else 1

        return {
            'avg_contexts_retrieved': total_contexts / num_questions,
            'avg_context_length': total_length / max(1, total_contexts),
            'questions_with_contexts': questions_with_contexts
        }