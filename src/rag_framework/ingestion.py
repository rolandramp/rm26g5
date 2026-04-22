"""Corpus ingestion module using LangChain document loaders."""

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_core.documents import Document
from typing import List, Dict, Any
import os


class CorpusIngestion:
    """Handles loading and preprocessing of documents for RAG."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def load_documents(self) -> List[Document]:
        """Load documents based on configuration."""
        loader_type = self.config.get('loader_type', 'directory')
        path = self.config.get('path', './data')

        if loader_type == 'directory':
            # Load all files in directory
            loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader)
            documents = loader.load()
        elif loader_type == 'text':
            loader = TextLoader(path)
            documents = loader.load()
        elif loader_type == 'pdf':
            loader = PyPDFLoader(path)
            documents = loader.load()
        else:
            raise ValueError(f"Unsupported loader type: {loader_type}")

        return documents