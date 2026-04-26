"""
Retriever Configuration Module

This module configures how we search the vector store for relevant documents.

WHAT IS A RETRIEVER?
The retriever is the component that, given a query (question), finds the
most relevant documents from our vector store. It's the "R" in "RAG" -
Retrieval-Augmented Generation.

The key parameter here is 'k' - the number of documents to retrieve.
- If k is too small, we might miss relevant context
- If k is too large, we might overwhelm the language model with too much context
- The optimal value depends on your chunk size and the model's context window

WHY IS THIS IMPORTANT?
The retriever determines what context the language model sees. If it retrieves
irrelevant documents, the model will give irrelevant answers. This is often
the biggest source of RAG system failures!
"""

from langchain_core.vectorstores import VectorStoreRetriever
from typing import Dict, Any


class RetrieverConfig:
    """
    Configurable retriever for vector store searches.

    This class wraps the complexity of setting up a retriever and makes
    it easy to configure via our YAML config file.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration.

        Args:
            config: Dictionary with 'k' (number of results to retrieve)
        """
        self.config = config

    def get_retriever(self, vector_store):
        """
        Create a retriever from the vector store.

        Args:
            vector_store: A VectorStore instance (like Chroma)

        Returns:
            A retriever that can be used in a LangChain chain
        """
        k = self.config.get('k', 5)  # Default: retrieve top 5 results
        return vector_store.as_retriever(search_kwargs={"k": k})