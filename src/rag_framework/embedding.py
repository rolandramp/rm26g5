"""Embedding model integrations."""

from langchain_openai import OpenAIEmbeddings
from typing import Dict, Any


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
            return OpenAIEmbeddings(
                model=self.config.get('model_name', 'text-embedding-ada-002'),
                api_key=api_key,
                base_url=base_url
            )
        else:
            raise ValueError(f"Unsupported embedding model: {model_type}")