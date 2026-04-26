"""
Corpus Ingestion Module

This module handles loading documents and ground truth data for RAG evaluation.

In RAG systems, we need two main types of data:
1. The corpus - the documents we can retrieve from (the "knowledge base")
2. The Q&A pairs - questions with known correct answers (the "ground truth")

Having ground truth is essential for evaluation - it lets us compare what
our system produces against known correct answers.

This module currently supports loading from HuggingFace datasets, which is
a popular format for sharing benchmark datasets in the ML community.
"""

from langchain_core.documents import Document
from typing import Dict, Any, Tuple
from datasets import load_dataset
import os
import pandas as pd


class CorpusIngestion:
    """
    Handles loading and preprocessing of documents for RAG.

    This class is responsible for getting the raw data into our system.
    It knows how to read from different sources and format the data
    in the way our pipeline expects.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration.

        Args:
            config: Dictionary containing ingestion settings, particularly
                   the 'dataset' key if loading from HuggingFace
        """
        self.config = config

    def load_documents(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load documents and ground truth Q&A pairs.

        This is the main entry point - call this to get the data you need.

        Returns:
            A tuple of two DataFrames:
            - qna_df: Questions with reference answers (columns: question, answer, relevant_passage_ids, id)
            - corpus_df: The document passages we can search through (columns: passage, id)

        The Q&A DataFrame is our "ground truth" - it contains questions and
        the correct answers, which we use to evaluate whether our RAG system
        is working correctly.

        The corpus DataFrame is our knowledge base - these are the documents
        that will be chunked, embedded, and stored in the vector database.
        """
        # Check if we're configured to load from a HuggingFace dataset
        # HuggingFace datasets are a common way to share benchmark data
        if 'dataset' in self.config and self.config.get('dataset'):
            ds_name = self.config.get('dataset')

            # Get configuration for the Q&A and corpus subsets
            # These might be named differently depending on the dataset
            ds_qna_config = self.config.get('dataset_qna_config', self.config.get('dataset_config'))
            ds_corpus_config = self.config.get('dataset_corpus_config', self.config.get('dataset_config'))

            # Set up caching to avoid re-downloading every time
            cache_dir = self.config.get('dataset_cache_dir', './cache/datasets')
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)

            # Load both the Q&A and corpus splits from HuggingFace
            # Using load_dataset() from the 'datasets' library
            ds_qna = load_dataset(str(ds_name), name=ds_qna_config, cache_dir=cache_dir)
            ds_corpus = load_dataset(str(ds_name), name=ds_corpus_config, cache_dir=cache_dir)

            # Validate that the expected splits exist
            # We need 'test' for Q&A and 'passages' for the corpus
            if 'test' not in ds_qna:
                raise ValueError("Expected QnA dataset to contain a 'test' split.")
            if 'passages' not in ds_corpus:
                raise ValueError("Expected corpus dataset to contain a 'passages' split.")

            # Get the actual data splits
            qna_split = ds_qna['test']
            corpus_split = ds_corpus['passages']

            # Define what columns we need - this is our contract with the dataset
            required_qna_columns = ['question', 'answer', 'relevant_passage_ids', 'id']
            required_corpus_columns = ['passage', 'id']

            # Convert to pandas DataFrames for easier manipulation
            qna_df: pd.DataFrame = qna_split.to_pandas()
            corpus_df: pd.DataFrame = corpus_split.to_pandas()

            # Validate that the required columns actually exist
            # This catches errors early with clear messages
            missing_qna = set(required_qna_columns) - set(qna_df.columns)
            if missing_qna:
                raise ValueError(f"QnA dataset missing columns: {missing_qna}")

            missing_corpus = set(required_corpus_columns) - set(corpus_df.columns)
            if missing_corpus:
                raise ValueError(f"Corpus dataset missing columns: {missing_corpus}")

            # Select only the columns we need and ensure consistent ordering
            # This makes downstream code simpler - we know exactly what we have
            qna_df = qna_df[required_qna_columns]
            corpus_df = corpus_df[required_corpus_columns]

            return qna_df, corpus_df

        # If no dataset is configured, return empty DataFrames
        # This allows the code to continue running even without data
        # (useful for testing or debugging specific components)
        return pd.DataFrame(columns=['question', 'answer', 'relevant_passage_ids', 'id']), pd.DataFrame(columns=['passage', 'id'])
