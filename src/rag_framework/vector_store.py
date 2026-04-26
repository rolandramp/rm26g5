"""
Vector Store Module

This module handles the storage and retrieval of document embeddings.

WHAT IS A VECTOR STORE?
A vector store is a specialized database designed to:
1. Store numerical vectors (embeddings) efficiently
2. Perform fast similarity searches
3. Return the most similar items to a given query vector

When you "index" documents, you're:
1. Converting each chunk to an embedding vector
2. Storing those vectors in the vector store

When you "retrieve," you're:
1. Converting the query to an embedding
2. Finding the stored vectors most similar to your query

Chroma is a popular open-source vector store that's lightweight and
easy to use. It stores embeddings on disk, so they persist between runs.

WHY IS THIS IMPORTANT?
The vector store is the heart of retrieval in RAG. The quality of your
search results directly impacts how well the language model can answer
questions - if it gets bad context, it gives bad answers.
"""

from langchain_community.vectorstores import VectorStore
from langchain_chroma import Chroma
from typing import Dict, Any


class VectorStoreManager:
    """
    Manages vector store operations.

    This class provides a simple interface for creating and interacting
    with vector databases. It currently supports Chroma, but could be
    extended to support others (like Pinecone, Weaviate, Milvus, etc.)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration.

        Args:
            config: Dictionary with 'store_type', 'persist_directory', etc.
        """
        self.config = config

    def get_vector_store(self, embedding) -> VectorStore:
        """
        Get a vector store instance.

        This creates a vector store backed by the specified embedding function.
        If the store already exists at the persist_directory, it loads it.
        Otherwise, it creates a new one.

        Args:
            embedding: An embedding function (like from EmbeddingModel)

        Returns:
            A VectorStore instance that can add_documents() and as_retriever()
        """
        store_type = self.config.get('store_type', 'chroma')
        persist_directory = self.config.get('persist_directory', './chroma_db')

        if store_type == 'chroma':
            # Chroma is our supported vector store
            # The persist_directory is where it saves data to disk
            return Chroma(
                embedding_function=embedding,
                persist_directory=persist_directory
            )
        else:
            raise ValueError(f"Unsupported vector store: {store_type}")