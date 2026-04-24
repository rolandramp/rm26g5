"""Embedding model integrations."""

from langchain_openai import OpenAIEmbeddings
from typing import Dict, Any, List
from openai import BadRequestError


class OllamaCompatibleEmbeddings(OpenAIEmbeddings):
    """OpenAI-compatible embeddings with Ollama fallback behavior."""

    def __init__(self, **kwargs):
        super().__init__(check_embedding_ctx_length=False, **kwargs)

    def embed_documents(self, texts: List[str], chunk_size: int = None) -> List[List[float]]:
        normalized_texts = [text if isinstance(text, str) else str(text) for text in texts]
        try:
            return super().embed_documents(normalized_texts, chunk_size=chunk_size)
        except BadRequestError as exc:
            if "invalid input type" not in str(exc).lower():
                raise
            embeddings: List[List[float]] = []
            client_kwargs = {**self._invocation_params}
            chunk_size_ = chunk_size or self.chunk_size
            for i in range(0, len(normalized_texts), chunk_size_):
                response = self.client.create(
                    input=normalized_texts[i : i + chunk_size_], **client_kwargs
                )
                if not isinstance(response, dict):
                    response = response.model_dump()
                embeddings.extend(r["embedding"] for r in response["data"])
            return embeddings


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
