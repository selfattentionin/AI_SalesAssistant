# LG Refrigerator Sales Assistant

This project is a complete end-to-end retrieval-augmented generation (RAG) chatbot for LG refrigerator sales support.

## Features

- Reads PDF documents from a local directory
- Builds a FAISS vector database from the PDFs
- Uses Gemini 1.5 Flash via Google Generative AI
- Answers customer questions with product-grounded responses
- Includes a Streamlit frontend ready for Streamlit Community Cloud
- Provides a FastAPI wrapper for API-based integrations

## Project structure

- `app.py` — main Streamlit application
- `api.py` — Gemini-backed RAG assistant logic
- `assist.py` — PDF ingestion and vector store creation
- `fast_api.py` — minimal REST API wrapper
- `frontend.py` — compatibility wrapper for the Streamlit app
- `documents/` — folder for PDF product manuals and brochures
- `.vectorstore/` — generated FAISS vector index

## Setup

1. Install dependencies:
   `pip install -r requirements.txt`
2. Create a `.env` file with:
   `GEMINI_API_KEY=your-google-api-key`
3. Add PDF files into the `documents/` folder.
4. Run the app:
   `streamlit run app.py`

## Streamlit Cloud deployment

1. Push this repository to GitHub.
2. In Streamlit Cloud, create a new app from the repo.
3. Set the secret `GEMINI_API_KEY` under app secrets.
4. Set the main file path to `app.py`.
5. Deploy.

## Notes

If the `documents/` folder is empty, the app automatically creates a starter LG refrigerator product guide PDF so the assistant can initialize immediately.
