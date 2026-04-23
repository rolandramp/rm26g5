"""Embedding model integrations."""

from langchain_openai import OpenAIEmbeddings
from typing import Dict, Any
from openai import BadRequestError


class OllamaCompatibleEmbeddings(OpenAIEmbeddings):
    """OpenAI-compatible embeddings with Ollama fallback behavior."""

    def embed_documents(self, texts, chunk_size=None):
        normalized_texts = [text if isinstance(text, str) else str(text) for text in texts]
        try:
            return super().embed_documents(normalized_texts, chunk_size=chunk_size)
        except BadRequestError as exc:
            message = str(exc).lower()
            if "invalid input type" not in message:
                raise
            return [self.embed_query(text) for text in normalized_texts]


class EmbeddingModel:
    """Configurable embedding models."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def get_embedding(self):
        """Get the embedding model instance."""
        model_type = self.config.get('model_type', 'openai')
        api_key = self.config.get('api_key')
        base_url = self.config.get('base_url')

        if model_type == 'openai':
            return OllamaCompatibleEmbeddings(
                model=self.config.get('model_name', 'text-embedding-ada-002'),
                api_key=api_key,
                base_url=base_url
            )
        else:
            raise ValueError(f"Unsupported embedding model: {model_type}")
