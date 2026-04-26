"""
Chunking Module

This module handles the critical task of splitting large documents into
smaller, manageable chunks.

WHY IS CHUNKING IMPORTANT?
In RAG systems, we have a fundamental constraint: language models have a
maximum context length (how many tokens they can process at once). We can't
just dump an entire document library into the model.

Instead, we:
1. Split documents into chunks (typically 500-2000 characters each)
2. Embed each chunk as a vector
3. When a user asks a question, find the most relevant chunks
4. Feed only those chunks to the language model

The chunking strategy significantly impacts retrieval quality. Too small,
and you lose context. Too big, and you might include irrelevant information
or hit token limits.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict, Any


class ChunkingStrategy:
    """
    Configurable text splitting strategies for documents.

    This class provides different ways to break up text into chunks.
    The strategy and parameters can dramatically affect downstream
    retrieval quality, so this is an important thing to experiment with.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration.

        Args:
            config: Dictionary with keys like 'strategy', 'chunk_size', 'chunk_overlap'
        """
        self.config = config

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.

        This is the main method - you give it full documents, and it returns
        a list of smaller document chunks that can be embedded and searched.

        Args:
            documents: List of LangChain Document objects to split

        Returns:
            List of smaller Document objects (chunks)
        """
        # Get configuration values with sensible defaults
        strategy = self.config.get('strategy', 'recursive')
        chunk_size = self.config.get('chunk_size', 1000)
        chunk_overlap = self.config.get('chunk_overlap', 200)

        # The 'recursive' strategy is the most commonly used and generally
        # works well for most text. It tries different separators in order
        # of priority:
        # 1. Double newlines (paragraphs)
        # 2. Single newlines (lines)
        # 3. Spaces (sentences)
        # 4. Individual characters (as last resort)
        #
        # The overlap is important - it ensures that context isn't lost
        # at chunk boundaries. If a chunk ends in the middle of a sentence,
        # the next chunk starts with some overlap so we don't lose meaning.
        if strategy == 'recursive':
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,      # Target size for each chunk in characters
                chunk_overlap=chunk_overlap, # How much overlap between chunks
                separators=["\n\n", "\n", " ", ""]  # Try these in order
            )
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")

        # Let LangChain's splitter do the actual work
        return splitter.split_documents(documents)