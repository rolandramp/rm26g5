"""Evaluation metrics for RAG pipelines."""

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
    """Evaluates RAG pipeline performance."""

    def __init__(self, config: Dict[str, Any], evaluator_config: Optional[Dict[str, Any]] = None, generator_config: Optional[Dict[str, Any]] = None):
        self.config = config
        self.evaluator_config = evaluator_config or {}
        self.generator_config = generator_config or {}

    def _get_evaluator_llm(self):
        """Get the evaluator LLM for RAGAS."""
        model_name = self.generator_config.get('model_name', 'gpt-3.5-turbo')
        api_key = self.generator_config.get('api_key', 'ollama')
        base_url = self.generator_config.get('base_url')

        print(f"[DEBUG] Creating evaluator LLM: model={model_name}, base_url={base_url}")

        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0
        )

    def _get_evaluator_embeddings(self):
        """Get the evaluator embeddings for RAGAS."""
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
                    latencies, costs):
        """Compute all metrics for a batch of results."""
        results = {}

        formatted_contexts = []
        for ctx_list in contexts:
            if not ctx_list:
                formatted_contexts.append([])
            elif isinstance(ctx_list[0], str):
                formatted_contexts.append(ctx_list)
            else:
                formatted_contexts.append(ctx_list)

        data = {
            "question": questions,
            "answer": generated_answers,
            "contexts": formatted_contexts,
            "ground_truth": reference_answers
        }
        dataset = Dataset.from_dict(data)

        try:
            print("[DEBUG] Setting up evaluator LLM for RAGAS...")
            evaluator_llm = self._get_evaluator_llm()
            evaluator_embeddings = self._get_evaluator_embeddings()

            print("[DEBUG] Running RAGAS evaluation with metrics...")
            ragas_metrics = [
                SemanticSimilarity(),
                AnswerCorrectness(),
                ContextPrecision(),
                BleuScore(),
                RougeScore(),
                Faithfulness()
            ]
            ragas_results = evaluate(
                dataset,
                metrics=ragas_metrics,
                llm=evaluator_llm,
                embeddings=evaluator_embeddings
            )
            results['context_precision'] = ragas_results['context_precision']
            results['faithfulness'] = ragas_results['faithfulness']
            results['semantic_similarity'] = ragas_results['semantic_similarity']
            results['answer_correctness'] = ragas_results['answer_correctness']
            results['bleu_score'] = ragas_results['bleu_score']
            results['rouge_score'] = ragas_results['rouge_score']
        except Exception as e:
            print(f"[WARNING] RAGAS evaluation failed: {e}")
            print("[DEBUG] Falling back to basic metrics only")

        results['lexical_similarity_per_question'] = self._compute_lexical_similarity(generated_answers, reference_answers)
        results['lexical_similarity_avg'] = float(np.mean(results['lexical_similarity_per_question'])) if results['lexical_similarity_per_question'] else 0.0
        results['avg_latency'] = float(np.mean(latencies)) if latencies else 0.0
        results['total_cost'] = sum(costs)
        results['retrieval_quality'] = self._compute_retrieval_quality(formatted_contexts, reference_answers)

        return results

    def _compute_lexical_similarity(self, generated, reference):
        """Compute simple word overlap similarity."""
        scores = []
        for gen, ref in zip(generated, reference):
            gen_words = set(gen.lower().split()) if gen else set()
            ref_words = set(ref.lower().split()) if ref else set()
            if gen_words and ref_words:
                overlap = len(gen_words & ref_words) / len(gen_words | ref_words)
                scores.append(overlap)
            else:
                scores.append(0.0)
        return scores  # Return per-question list

    def _compute_retrieval_quality(self, contexts, reference_answers):
        """Compute retrieval metrics from actual retrieved contexts."""
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