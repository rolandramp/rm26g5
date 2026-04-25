"""LLM answer generation for RAG."""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any, List
from langchain_core.documents import Document


class RAGGenerator:
    """RAG answer generation using LLMs."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._last_retrieved_contexts: List[str] = []

    def get_last_retrieved_contexts(self) -> List[str]:
        """Get the contexts from the last retrieval."""
        return self._last_retrieved_contexts

    def clear_last_retrieved_contexts(self) -> None:
        """Clear the stored contexts."""
        self._last_retrieved_contexts = []

    def get_chain(self, retriever):
        """Build the RAG chain that captures contexts."""
        llm = ChatOpenAI(
            model=self.config.get('model_name', 'gpt-3.5-turbo'),
            api_key=self.config.get('api_key') or 'ollama',
            base_url=self.config.get('base_url')
        )

        template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
        prompt = ChatPromptTemplate.from_template(template)

        def extract_contexts(inputs):
            """Extract and store retrieved contexts before they're passed to the prompt."""
            docs = inputs.get("context", [])
            if isinstance(docs, list):
                contexts = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in docs]
            else:
                contexts = [str(docs)]
            self._last_retrieved_contexts = contexts
            return inputs

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | RunnableLambda(extract_contexts)
            | {"context": RunnableLambda(lambda x: x["context"]), "question": RunnableLambda(lambda x: x["question"])}
            | prompt
            | llm
            | StrOutputParser()
        )

        return chain