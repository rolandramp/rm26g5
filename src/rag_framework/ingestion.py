"""Corpus ingestion module using LangChain document loaders."""

from langchain_core.documents import Document
from typing import Dict, Any, Tuple
from datasets import load_dataset
import os
import pandas as pd


class CorpusIngestion:
    """Handles loading and preprocessing of documents for RAG."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def load_documents(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load documents and ground truth QnA from dataset.

        Returns:
            Tuple of (qna_df, corpus_df) where:
                - qna_df: pd.DataFrame with columns ['question','answer','relevant_passage_ids','id']
                - corpus_df: pd.DataFrame with columns ['passage','id']
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

            required_qna_columns = ['question', 'answer', 'relevant_passage_ids', 'id']
            required_corpus_columns = ['passage', 'id']

            qna_df: pd.DataFrame = qna_split.to_pandas()
            corpus_df: pd.DataFrame = corpus_split.to_pandas()

            # Validate required columns exist in the loaded splits
            missing_qna = set(required_qna_columns) - set(qna_df.columns)
            if missing_qna:
                raise ValueError(f"QnA dataset missing columns: {missing_qna}")

            missing_corpus = set(required_corpus_columns) - set(corpus_df.columns)
            if missing_corpus:
                raise ValueError(f"Corpus dataset missing columns: {missing_corpus}")

            # Select and order columns explicitly
            qna_df = qna_df[required_qna_columns]
            corpus_df = corpus_df[required_corpus_columns]

            return qna_df, corpus_df

        # If dataset not configured or other branch, return empty DataFrames
        return pd.DataFrame(columns=['question', 'answer', 'relevant_passage_ids', 'id']), pd.DataFrame(columns=['passage', 'id'])
