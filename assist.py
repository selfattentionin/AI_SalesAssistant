"""Utilities for building a PDF-backed LG refrigerator RAG system."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


DEFAULT_DOCUMENTS_DIR = "documents"
DEFAULT_VECTOR_STORE_DIR = ".vectorstore"


def _create_sample_lg_refrigerator_pdf(pdf_path: Path) -> None:
    """Generate a sample LG refrigerator product brochure PDF for first-time setup."""
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    story = [Paragraph("LG Refrigerator Product Guide", styles["Title"]), Spacer(1, 18)]

    body = [
        "LG refrigerators are designed for efficient cooling, quiet operation, and premium food care.",
        "Key models include the LG French Door refrigerator with Door-in-Door convenience, twin cooling systems, and smart diagnosis.",
        "The LG Top Freezer range is ideal for budget-conscious homes with dependable performance and large storage space.",
        "LG InstaView refrigerators use a glass panel that illuminates when tapped, allowing quick access without opening the door.",
        "For family households, premium LG side-by-side models provide flexible storage, water dispensers, and energy efficiency.",
        "Sales guidance: recommend capacity based on household size, kitchen space, energy savings, and smart features.",
        "Common customer questions include capacity, noise, freezer space, and whether features like Water+Ice dispensers or InstaView are worth the upgrade.",
        "LG refrigerators typically emphasize inverter compressors, hygiene-focused features, and premium finishing that appeals to modern households."
    ]
    for text in body:
        story.append(Paragraph(text, styles["BodyText"]))
        story.append(Spacer(1, 10))

    doc.build(story)


def ensure_documents_directory(directory: str = DEFAULT_DOCUMENTS_DIR) -> Path:
    """Create the PDF directory and a starter LG guide if it is empty."""
    docs_dir = Path(directory)
    docs_dir.mkdir(parents=True, exist_ok=True)

    if not any(docs_dir.glob("*.pdf")):
        sample_path = docs_dir / "lg_refrigerator_sales_guide.pdf"
        _create_sample_lg_refrigerator_pdf(sample_path)

    return docs_dir


def load_documents_from_directory(directory: str = DEFAULT_DOCUMENTS_DIR) -> List[Document]:
    """Load all PDF files from a directory into LangChain Documents."""
    docs_dir = ensure_documents_directory(directory)
    documents: List[Document] = []

    for pdf_path in sorted(docs_dir.glob("*.pdf")):
        try:
            loader = PyPDFLoader(str(pdf_path))
            file_documents = loader.load()
            if file_documents:
                for doc in file_documents:
                    doc.metadata.setdefault("source", str(pdf_path))
                documents.extend(file_documents)
                continue
        except Exception:
            pass

        try:
            raw_text = pdf_path.read_bytes().decode("utf-8", errors="ignore")
        except Exception:
            raw_text = f"LG refrigerator product content for {pdf_path.name}."

        if not raw_text.strip():
            raw_text = f"LG refrigerator product content for {pdf_path.name}."

        documents.append(
            Document(
                page_content=f"Source: {pdf_path.name}\n{raw_text}",
                metadata={"source": str(pdf_path)},
            )
        )

    if not documents:
        raise ValueError(f"No PDF files were found in the directory: {docs_dir}")

    return documents


def build_vector_store(pdf_dir: str = DEFAULT_DOCUMENTS_DIR, persist_dir: str = DEFAULT_VECTOR_STORE_DIR):
    """Create and persist a FAISS vector store from PDFs in the configured directory."""
    docs_dir = ensure_documents_directory(pdf_dir)
    docs = load_documents_from_directory(str(docs_dir))

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)

    save_path = Path(persist_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(save_path))
    return vector_store


def load_vector_store(persist_dir: str = DEFAULT_VECTOR_STORE_DIR):
    """Load an existing FAISS index from disk."""
    save_path = Path(persist_dir)
    if not save_path.exists():
        raise FileNotFoundError(f"No vector store exists at: {save_path}")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(str(save_path), embeddings, allow_dangerous_deserialization=True)


def get_catalog_stats() -> dict:
    """Return metadata about the sales knowledge base."""
    docs_dir = ensure_documents_directory(DEFAULT_DOCUMENTS_DIR)
    pdf_files = sorted(docs_dir.glob("*.pdf"))
    return {
        "documents_directory": str(docs_dir),
        "document_count": len(pdf_files),
        "pdf_files": [p.name for p in pdf_files],
        "vector_store_directory": DEFAULT_VECTOR_STORE_DIR,
    }
