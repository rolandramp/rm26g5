"""Corpus ingestion module using LangChain document loaders."""

from langchain_core.documents import Document
from typing import List, Dict, Any, Tuple
from datasets import load_dataset
import os


class CorpusIngestion:
    """Handles loading and preprocessing of documents for RAG."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def load_documents(self) -> Tuple[List[Document], List[Dict[str, Any]]]:
        """Load documents and ground truth QnA from dataset.

        Returns:
            Tuple of (documents, ground_truth) where:
                - documents: List of Document objects from corpus
                - ground_truth: List of dicts with 'question', 'answer', 'id' keys
        """
        # If a Hugging Face dataset is specified, prefer that.
        if 'dataset' in self.config and self.config.get('dataset'):
            ds_name = self.config.get('dataset')
            ds_qna_config = self.config.get('dataset_qna_config', self.config.get('dataset_config'))
            ds_corpus_config = self.config.get('dataset_corpus_config', self.config.get('dataset_config'))

            cache_dir = self.config.get('dataset_cache_dir', './cache/datasets')
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)

            ds_qna = load_dataset(str(ds_name), name=ds_qna_config, cache_dir=cache_dir)
            ds_corpus = load_dataset(str(ds_name), name=ds_corpus_config, cache_dir=cache_dir)

            if 'test' not in ds_qna:
                raise ValueError("Expected QnA dataset to contain a 'test' split.")
            if 'passages' not in ds_corpus:
                raise ValueError("Expected corpus dataset to contain a 'passages' split.")

            qna_split = ds_qna['test']
            corpus_split = ds_corpus['passages']

            required_qna_columns = {'question', 'answer', 'relevant_passage_ids', 'id'}
            required_corpus_columns = {'passage', 'id'}

            if not required_qna_columns.issubset(set(qna_split.column_names)):
                missing = required_qna_columns.difference(set(qna_split.column_names))
                raise ValueError(f"QnA dataset is missing required columns: {sorted(missing)}")

            if not required_corpus_columns.issubset(set(corpus_split.column_names)):
                missing = required_corpus_columns.difference(set(corpus_split.column_names))
                raise ValueError(f"Corpus dataset is missing required columns: {sorted(missing)}")

            passage_references: Dict[str, List[Any]] = {}
            for qna_row in qna_split:
                qna_id = qna_row.get('id')
                relevant_ids = qna_row.get('relevant_passage_ids', []) or []
                for passage_id in relevant_ids:
                    key = str(passage_id)
                    if key not in passage_references:
                        passage_references[key] = []
                    passage_references[key].append(qna_id)

            documents: List[Document] = []
            for corpus_row in corpus_split:
                passage_text = corpus_row.get('passage', '')
                passage_id = corpus_row.get('id')

                if not isinstance(passage_text, str):
                    passage_text = str(passage_text)

                metadata = {
                    'source': 'hf_dataset_corpus',
                    'dataset': str(ds_name),
                    'id': passage_id,
                    'linked_qna_ids': passage_references.get(str(passage_id), []),
                }

                documents.append(Document(page_content=passage_text, metadata=metadata))

            ground_truth: List[Dict[str, Any]] = []
            for qna_row in qna_split:
                ground_truth.append({
                    'question': qna_row.get('question'),
                    'answer': qna_row.get('answer'),
                    'id': qna_row.get('id'),
                })

            return documents, ground_truth

     

        return [], []
