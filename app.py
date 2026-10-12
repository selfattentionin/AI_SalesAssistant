import os
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

from api import initialize_chatbot
from assist import build_vector_store, get_catalog_stats

load_dotenv()

st.set_page_config(
    page_title="SelfAttention.in | LG Refrigerator Sales Assistant",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Brand watermark for the application
st.markdown(
    """
    <style>
        .watermark {
            position: fixed;
            bottom: 20px;
            right: 25px;
            opacity: 0.14;
            font-size: 30px;
            font-weight: 800;
            color: #1d4ed8;
            pointer-events: none;
            z-index: 9999;
            letter-spacing: 1px;
            animation: pulse 3s ease-in-out infinite;
        }
        .brand-block {
            margin-bottom: 0.5rem;
        }
        .intern-tag {
            color: #475569;
            font-size: 0.95rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            margin-top: 0.25rem;
            text-transform: uppercase;
        }
        .hero-panel {
            background: linear-gradient(135deg, rgba(14,116,144,0.18), rgba(59,130,246,0.08), rgba(80,200,120,0.12));
            border: 1px solid rgba(148,163,184,0.3);
            border-radius: 22px;
            padding: 1.2rem 1.3rem;
            margin-bottom: 1rem;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.12);
            animation: fadeSlide 1.2s ease;
        }
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
            margin: 1rem 0 1.5rem;
        }
        .stat-card {
            background: linear-gradient(135deg, #0f172a, #1e3a8a);
            color: white;
            border-radius: 18px;
            padding: 1rem;
            min-height: 110px;
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.22);
            animation: floatCard 3s ease-in-out infinite;
        }
        .stat-card:nth-child(2) { animation-delay: 0.5s; }
        .stat-card:nth-child(3) { animation-delay: 1s; }
        .stat-label {
            opacity: 0.8;
            font-size: 0.8rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .stat-value {
            font-size: 2rem;
            font-weight: 800;
            margin-top: 0.3rem;
        }
        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin: 0.8rem 0 1rem;
        }
        .chip {
            background: linear-gradient(135deg, #dbeafe, #e0f2fe);
            color: #0f172a;
            border-radius: 999px;
            padding: 0.45rem 0.8rem;
            font-size: 0.8rem;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.12);
            animation: popIn 0.8s ease;
        }
        @keyframes floatCard {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-5px); }
        }
        @keyframes fadeSlide {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
            0%, 100% { opacity: 0.12; }
            50% { opacity: 0.22; }
        }
        @keyframes popIn {
            from { opacity: 0; transform: scale(0.96); }
            to { opacity: 1; transform: scale(1); }
        }
    </style>
    <div class="watermark">selfattention.in</div>
    """,
    unsafe_allow_html=True,
)


def get_api_key() -> Optional[str]:
    """Read Gemini API key from env or Streamlit secrets."""
    value = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if value:
        return value

    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None


def initialize_session_assistant() -> Optional[object]:
    key = get_api_key()
    if not key:
        return None
    if "assistant" not in st.session_state or st.session_state.get("assistant_key") != key:
        st.session_state.assistant = initialize_chatbot(api_key=key)
        st.session_state.assistant_key = key
    return st.session_state.assistant


def rebuild_knowledge_base() -> None:
    try:
        with st.spinner("Rebuilding the PDF vector database..."):
            build_vector_store("documents", ".vectorstore")
        st.success("Knowledge base rebuilt successfully.")
    except Exception as exc:  # pragma: no cover
        st.error(f"Unable to rebuild the knowledge base: {exc}")


def render_sidebar() -> None:
    with st.sidebar:
        st.title("LG Refrigerator Sales Bot")
        st.caption("PDF-driven RAG assistant for LG refrigerators")

        api_key = st.text_input("Gemini API key", type="password", value=get_api_key() or "")
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
            st.session_state["entered_api_key"] = api_key

        if st.button("Rebuild vector store", use_container_width=True):
            rebuild_knowledge_base()

        st.divider()
        st.subheader("Knowledge base")
        try:
            stats = get_catalog_stats()
            st.metric("PDF files", stats["document_count"])
            st.write(stats["documents_directory"])
            st.write("Stored vector index: .vectorstore")
        except Exception:
            st.info("Add PDF files to the documents folder to populate the RAG database.")

        st.divider()
        st.subheader("Prompt ideas")
        suggestions = [
            "What LG refrigerator is best for a family of 4?",
            "Suggest a quiet, energy-efficient LG model under $1000.",
            "Compare French door and side-by-side LG refrigerators.",
            "I need a compact LG model for a small apartment.",
        ]
        for suggestion in suggestions:
            if st.button(suggestion, key=suggestion, use_container_width=True):
                st.session_state["pending_prompt"] = suggestion


def format_sources(sources):
    if not sources:
        return "No source file references were returned."
    return "\n".join(f"- {source}" for source in sources)


def main() -> None:
    render_sidebar()

    st.markdown(
        """
        <div class="hero-panel">
            <div class="brand-block">
                <h1 style='margin-bottom: 0; color: #0f172a;'>Intern Project by Students at selfattention.in</h1>
                <div class="intern-tag">LG Refrigerator Sales Assistant</div>
            </div>
            <div class="chip-row">
                <span class="chip">Gemini RAG</span>
                <span class="chip">PDF Search</span>
                <span class="chip">Sales Guidance</span>
                <span class="chip">LG Refrigerators</span>
            </div>
            <p style='margin: 0; color: #334155; font-size: 1rem;'>This assistant answers customer questions using product PDFs, FAISS vector search, and Gemini-powered recommendations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-label">Documents</div>
                <div class="stat-value">PDF</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Vector DB</div>
                <div class="stat-value">FAISS</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">AI Model</div>
                <div class="stat-value">Gemini</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "pending_prompt" in st.session_state and st.session_state["pending_prompt"]:
        prompt = st.session_state["pending_prompt"]
        st.session_state["pending_prompt"] = ""
    else:
        prompt = st.chat_input("Ask about LG refrigerators, sizes, pricing, or features...")

    if get_api_key() is None:
        st.warning("Add your Gemini API key in the sidebar or set GEMINI_API_KEY in a .env file before asking questions.")
        return

    assistant = initialize_session_assistant()
    if assistant is None:
        st.error("The assistant could not be initialized. Please validate the Gemini API key.")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("role") == "assistant":
                if message.get("sources"):
                    with st.expander("Source documents"):
                        st.code(format_sources(message["sources"]))
                if message.get("observability"):
                    metrics = message["observability"]
                    with st.expander("Observability"):
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Latency", f"{metrics.get('latency_ms', 0)} ms")
                        col2.metric("Input tokens", metrics.get("input_tokens", 0))
                        col3.metric("Output tokens", metrics.get("output_tokens", 0))
                        col4.metric("Chunks retrieved", metrics.get("retrieved_chunks", 0))

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Retrieving relevant LG product information..."):
            result = assistant.ask(prompt)

        answer = result.get("answer", "I could not find a suitable answer in the available documentation.")
        sources = result.get("sources", [])
        observability = result.get("observability", {})

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "observability": observability,
        })

        with st.chat_message("assistant"):
            st.markdown(answer)
            if sources:
                with st.expander("Source documents"):
                    st.code(format_sources(sources))
            if observability:
                with st.expander("Observability"):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Latency", f"{observability.get('latency_ms', 0)} ms")
                    col2.metric("Input tokens", observability.get("input_tokens", 0))
                    col3.metric("Output tokens", observability.get("output_tokens", 0))
                    col4.metric("Chunks retrieved", observability.get("retrieved_chunks", 0))

        st.rerun()


if __name__ == "__main__":
    main()
