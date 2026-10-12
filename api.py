"""Gemini-backed LG refrigerator sales assistant for PDF-based RAG."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from dotenv import load_dotenv

try:
    #from langchain.chains import RetrievalQA
    from langchain_classic.chains import RetrievalQA
except ImportError:  # LangChain 0.3+/newer package layout
    from langchain_community.chains import RetrievalQA

from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from assist import build_vector_store, ensure_documents_directory, get_catalog_stats, load_vector_store

load_dotenv()


def get_api_key_from_env() -> str:
    """Read the Gemini API key from environment variables or Streamlit secrets in deployment."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Add it to your .env file or Streamlit Cloud secrets before running the app."
        )
    return api_key


def get_security_features() -> Dict[str, list[str]]:
    """Describe deployment and safety guardrails for the assistant."""
    return {
        "RAG safeguards": [
            "Grounded answers from local PDF knowledge sources",
            "Vector retrieval limited to uploaded product documentation",
            "Source citations included with responses",
        ],
        "LLM reliability": [
            "Gemini model used for answer generation",
            "Fallback when no relevant context is available",
            "Structured sales guidance based on the product docs",
        ],
        "Deployment posture": [
            "Streamlit Cloud friendly configuration",
            "Environment-based API key management",
            "Knowledge base stored in a reusable vector index",
        ],
    }


def _build_sales_prompt() -> PromptTemplate:
    return PromptTemplate(
        template="""You are Alex, an expert sales assistant for LG refrigerators.
You help buyers choose the right LG fridge based on household size, kitchen space, energy efficiency, budget, and feature preferences.
Use only the product information in the context. If the exact detail is missing, say so briefly and give a practical recommendation based on general LG refrigerator guidance.

Required response format:
- Write exactly 2 complete sentences.
- Sentence 1: directly answer the user's question in a clear, sales-friendly description.
- Sentence 2: add one short recommendation or reason, such as capacity fit, feature benefit, or best model for the situation.
- Keep it concise, natural, and descriptive.
- Do not use bullet points, lists, or long paragraphs.

Context:
{context}

Question:
{question}

Answer:""",
        input_variables=["context", "question"],
    )


class LGRefrigeratorSalesAssistant:
    """A full PDF retrieval + generation assistant for LG refrigerator sales."""

    def __init__(self, documents_dir: str = "documents", vector_store_dir: str = ".vectorstore", api_key: Optional[str] = None):
        self.documents_dir = documents_dir
        self.vector_store_dir = vector_store_dir
        self.api_key = api_key or get_api_key_from_env()

        ensure_documents_directory(self.documents_dir)
        try:
            self.vector_store = load_vector_store(self.vector_store_dir)
        except FileNotFoundError:
            self.vector_store = build_vector_store(self.documents_dir, self.vector_store_dir)

        model_candidates = [
            "gemini-3.8-flash",
            # "gemini-1.5-flash",
            # "gemini-1.5-flash-latest",
        ]

        self.llm = None
        last_error = None
        for model_name in model_candidates:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=self.api_key,
                    temperature=0.2,
                    max_output_tokens=512,
                )
                break
            except Exception as exc:  # pragma: no cover - model availability is runtime-dependent
                last_error = exc

        if self.llm is None:
            raise RuntimeError(f"No supported Gemini Flash model was available for this account. Last error: {last_error}")

        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": _build_sales_prompt()},
            return_source_documents=True,
        )

    def ask(self, question: str) -> Dict[str, Any]:
        question = (question or "").strip()
        if not question:
            raise ValueError("Please enter a question about LG refrigerators.")

        start_time = time.perf_counter()
        result = self.chain.invoke({"query": question})
        latency_seconds = time.perf_counter() - start_time

        answer = (result.get("result", "") or "").strip()

        if answer:
            sentences = [s.strip() for s in answer.split(".") if s.strip()]
            if len(sentences) > 2:
                answer = ". ".join(sentences[:2]) + "."
            elif len(sentences) == 1:
                answer = sentences[0].strip() + "."

        sources = []
        for doc in result.get("source_documents", []):
            source = doc.metadata.get("source", "Unknown source")
            if source not in sources:
                sources.append(source)

        input_tokens = max(1, len(question.split()))
        output_tokens = max(1, len(answer.split()))
        observability = {
            "latency_ms": round(latency_seconds * 1000, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "retrieved_chunks": len(sources),
        }

        return {
            "answer": answer or "I could not find a direct answer in the product knowledge base. Please ask about capacity, budget, or feature preferences and I can suggest the best LG model.",
            "sources": sources,
            "observability": observability,
        }

    def chat(self, question: str) -> Dict[str, Any]:
        return self.ask(question)

    def clear_memory(self):
        """The RAG pipeline keeps no session memory; kept for compatibility."""
        return None


def initialize_chatbot(api_key: Optional[str] = None, documents_dir: str = "documents", vector_store_dir: str = ".vectorstore") -> LGRefrigeratorSalesAssistant:
    """Initialize the LG refrigerator sales assistant."""
    return LGRefrigeratorSalesAssistant(
        documents_dir=documents_dir,
        vector_store_dir=vector_store_dir,
        api_key=api_key,
    )
