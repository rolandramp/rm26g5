"""Retriever configurations."""

from langchain_core.vectorstores import VectorStoreRetriever
from typing import Dict, Any


class RetrieverConfig:
    """Configurable retrievers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def get_retriever(self, vector_store):
        """Get retriever instance."""
        k = self.config.get('k', 5)
        return vector_store.as_retriever(search_kwargs={"k": k})