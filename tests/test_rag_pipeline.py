import os
from pathlib import Path

from assist import build_vector_store, load_documents_from_directory


def test_load_documents_from_directory_reads_pdfs(tmp_path):
    pdf_dir = tmp_path / "documents"
    pdf_dir.mkdir()
    sample_pdf = pdf_dir / "sample.pdf"
    sample_pdf.write_bytes(b"%PDF-1.4\nThis is a fake PDF for tests.")

    docs = load_documents_from_directory(str(pdf_dir))

    assert len(docs) == 1
    assert "LG refrigerator" in docs[0].page_content.lower() or "fake pdf" in docs[0].page_content.lower()


def test_build_vector_store_creates_index(tmp_path):
    pdf_dir = tmp_path / "documents"
    pdf_dir.mkdir()
    sample_pdf = pdf_dir / "sample.pdf"
    sample_pdf.write_bytes(b"%PDF-1.4\nLG refrigerator sales brochure for family-size models.")

    store_path = tmp_path / "vector_index"
    vector_store = build_vector_store(str(pdf_dir), str(store_path))

    assert vector_store is not None
    assert store_path.exists()
