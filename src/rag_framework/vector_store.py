"""Vector database integrations."""

from langchain_community.vectorstores import VectorStore
from langchain_chroma import Chroma
from typing import Dict, Any


class VectorStoreManager:
    """Manages vector store operations."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def get_vector_store(self, embedding) -> VectorStore:
        """Get vector store instance."""
        store_type = self.config.get('store_type', 'chroma')
        persist_directory = self.config.get('persist_directory', './chroma_db')

        if store_type == 'chroma':
            return Chroma(
                embedding_function=embedding,
                persist_directory=persist_directory
            )
        else:
            raise ValueError(f"Unsupported vector store: {store_type}")