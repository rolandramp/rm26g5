"""
RAG Generator Module

This module handles the "generation" part of RAG - using a language model
to produce answers based on retrieved context.

THE RAG PIPELINE:
1. Retrieve: Find relevant documents from vector store
2. Augment: Put those documents in the prompt alongside the question
3. Generate: Have the LLM produce an answer using that context

This is where "Retrieval-Augmented Generation" happens! We don't just
let the model hallucinate - we force it to answer based on retrieved facts.

THE CHAIN:
We use LangChain's " LCEL" (LangChain Expression Language) to compose
the different steps together. The | operator chains components together,
passing output from one as input to the next.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any, List
from langchain_core.documents import Document


class RAGGenerator:
    """
    Handles answer generation for RAG using language models.

    This class is responsible for:
    1. Setting up the language model
    2. Creating the prompt template
    3. Building the chain that connects retrieval + generation
    4. Tracking what context was retrieved (for evaluation purposes)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration.

        Args:
            config: Dictionary with 'model_name', 'api_key', 'base_url'
        """
        self.config = config
        # Store the last retrieved contexts so we can evaluate later
        # This is crucial for understanding if retrieval worked well
        self._last_retrieved_contexts: List[str] = []

    def get_last_retrieved_contexts(self) -> List[str]:
        """
        Get the contexts that were retrieved for the last question.

        This is important for evaluation - we want to know not just
        what answer was generated, but what context it was based on.

        Returns:
            List of text chunks that were retrieved
        """
        return self._last_retrieved_contexts

    def clear_last_retrieved_contexts(self) -> None:
        """
        Clear stored contexts between questions.

        This should be called before processing each new question to
        ensure we're tracking the right contexts.
        """
        self._last_retrieved_contexts = []

    def get_chain(self, retriever):
        """
        Build the complete RAG chain.

        This creates a chain that:
        1. Takes a question as input
        2. Uses the retriever to find relevant documents
        3. Captures those documents for later analysis
        4. Formats them into a prompt
        5. Sends to the LLM for generation
        6. Returns just the text answer

        The chain uses LangChain's "pipe" syntax (|) to compose components.
        This is a declarative way to define processing pipelines.

        Args:
            retriever: A retriever instance (from RetrieverConfig)

        Returns:
            A Runnable chain that can be invoked with a question string
        """
        # Create the language model - supports both OpenAI and Ollama
        llm = ChatOpenAI(
            model=self.config.get('model_name', 'gpt-3.5-turbo'),
            api_key=self.config.get('api_key') or 'ollama',
            base_url=self.config.get('base_url')
        )

        # The prompt template tells the model how to use the context
        # We explicitly tell it to only use the provided context
        # This helps reduce hallucinations
        template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
        prompt = ChatPromptTemplate.from_template(template)

        def extract_contexts(inputs):
            """
            Capture retrieved contexts before they're consumed.

            This function is inserted into the chain to "spy" on what
            the retriever returned. We need this for evaluation - to
            know what context the answer was based on.
            """
            docs = inputs.get("context", [])
            if isinstance(docs, list):
                # Extract text from each document object
                contexts = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in docs]
            else:
                contexts = [str(docs)]
            # Store for later retrieval via get_last_retrieved_contexts()
            self._last_retrieved_contexts = contexts
            return inputs

        # Build the chain using LangChain Expression Language:
        # Step 1: Create dict with context (from retriever) and question (passed through)
        # Step 2: Extract and store the contexts
        # Step 3: Re-format for the prompt
        # Step 4: Apply the prompt template
        # Step 5: Call the LLM
        # Step 6: Parse the output as a string
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | RunnableLambda(extract_contexts)
            | {"context": RunnableLambda(lambda x: x["context"]), "question": RunnableLambda(lambda x: x["question"])}
            | prompt
            | llm
            | StrOutputParser()
        )

        return chain