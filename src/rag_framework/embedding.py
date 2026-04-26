"""
Embedding Model Integration

This module handles the conversion of text into numerical vectors (embeddings).

WHAT ARE EMBEDDINGS?
Embeddings are numerical representations of text that capture semantic meaning.
The key property is that similar texts have similar vectors - so when you want
to find documents relevant to a question, you can:

1. Convert the question into an embedding
2. Search for stored document embeddings that are "close" to the question

This is called "semantic search" - finding results based on meaning, not just
keyword matching. It's what makes RAG powerful!

The embedding model you choose affects:
- How well the model understands semantics
- The size of each vector (and thus storage requirements)
- The speed of similarity search
"""

from langchain_openai import OpenAIEmbeddings
from typing import Dict, Any, List
from openai import BadRequestError


class OllamaCompatibleEmbeddings(OpenAIEmbeddings):
    """
    OpenAI-compatible embeddings with support for Ollama models.

    Ollama is a tool that lets you run large language models locally on your
    own machine. This class extends OpenAIEmbeddings to work with Ollama's
    API, which is compatible with OpenAI's but runs locally.

    Why subclass instead of just using the regular OpenAI client?
    This handles a specific edge case where Ollama returns a different
    error type that we need to handle gracefully.
    """

    def __init__(self, **kwargs):
        # Pass check_embedding_ctx_length=False to avoid validation that
        # doesn't work well with Ollama's local models
        super().__init__(check_embedding_ctx_length=False, **kwargs)

    def embed_documents(self, texts: List[str], chunk_size: int = 1000):
        """
        Embed a list of texts.

        This overrides the parent method to handle an edge case where
        Ollama might return a different error format.

        Args:
            texts: List of text strings to embed
            chunk_size: How many texts to process at once

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        # First, ensure all inputs are strings
        # The embedding API expects strings, not other types
        normalized_texts = [text if isinstance(text, str) else str(text) for text in texts]

        try:
            # Try the normal approach first
            return super().embed_documents(normalized_texts, chunk_size=chunk_size)
        except BadRequestError as exc:
            # If we get an "invalid input type" error, try a different approach
            # This is a workaround for Ollama-specific behavior
            if "invalid input type" not in str(exc).lower():
                raise  # Re-raise if it's a different error

            # Fallback: process in smaller batches manually
            embeddings = []
            client_kwargs = {**self._invocation_params}
            chunk_size_ = chunk_size or self.chunk_size

            for i in range(0, len(normalized_texts), chunk_size_):
                batch = normalized_texts[i : i + chunk_size_]
                response = self.client.create(
                    input=batch, **client_kwargs
                )
                # Handle both dict and object response formats
                if not isinstance(response, dict):
                    response = response.model_dump()
                embeddings.extend(r["embedding"] for r in response["data"])
            return embeddings


class EmbeddingModel:
    """
    Factory class for creating embedding models.

    This class abstracts away the details of creating different embedding
    implementations based on configuration. You just say what "type" you
    want, and it creates the right object.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration.

        Args:
            config: Dictionary with 'model_type', 'model_name', 'api_key', etc.
        """
        self.config = config

    def get_embedding(self):
        """
        Get an embedding model instance.

        Returns:
            An embedding model that can embed_documents() and embed_query()
        """
        model_type = self.config.get('model_type', 'openai')
        api_key = self.config.get('api_key')
        base_url = self.config.get('base_url')

        if model_type == 'openai':
            # Use our Ollama-compatible class which works with both
            # OpenAI's API and local Ollama instances
            return OllamaCompatibleEmbeddings(
                model=self.config.get('model_name', 'text-embedding-ada-002'),
                api_key=self.config.get('api_key') or 'ollama',  # Default to 'ollama' for local
                base_url=self.config.get('base_url')
            )
        else:
            raise ValueError(f"Unsupported embedding model: {model_type}")