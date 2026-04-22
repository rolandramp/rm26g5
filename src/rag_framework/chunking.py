"""Chunking strategies for document splitting."""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict, Any


class ChunkingStrategy:
    """Configurable text splitting strategies."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        strategy = self.config.get('strategy', 'recursive')
        chunk_size = self.config.get('chunk_size', 1000)
        chunk_overlap = self.config.get('chunk_overlap', 200)

        if strategy == 'recursive':
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")

        return splitter.split_documents(documents)