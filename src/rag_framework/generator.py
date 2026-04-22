"""LLM answer generation for RAG."""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any


class RAGGenerator:
    """RAG answer generation using LLMs."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def get_chain(self, retriever):
        """Build the RAG chain."""
        llm = ChatOpenAI(
            model=self.config.get('model_name', 'gpt-3.5-turbo'),
            api_key=self.config.get('api_key'),
            base_url=self.config.get('base_url')
        )

        template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        return chain