import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

from chat import ask_once, load_index
from ingest import SUPPORTED_EXTENSIONS, build_index


def ensure_dirs() -> None:
    Path("docs").mkdir(parents=True, exist_ok=True)
    Path("docs/uploaded").mkdir(parents=True, exist_ok=True)
    Path("index").mkdir(parents=True, exist_ok=True)


def save_uploads(uploaded_files) -> list[str]:
    saved = []
    target_dir = Path("docs/uploaded")
    for f in uploaded_files:
        out_path = target_dir / f.name
        out_path.write_bytes(f.getbuffer())
        saved.append(str(out_path))
    return saved


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="RAG Chatbot", page_icon="📄", layout="wide")
    st.title("RAG Chatbot (Gemini + Your Documents)")
    st.caption("Grounded answers from your local files")

    ensure_dirs()

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        st.error("GEMINI_API_KEY is missing. Set it in rag_openai/.env first.")
        st.stop()

    genai.configure(api_key=api_key)
    model = os.getenv("GEMINI_MODEL", "auto")
    embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "auto")
    index_path = Path("index/index.json")

    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Top K Chunks", min_value=2, max_value=10, value=4)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.2, step=0.1)
        st.write(f"Chat model: `{model}`")
        st.write(f"Embedding model: `{embedding_model}`")
        st.write(f"Temperature: `{temperature}`")
        st.divider()
        st.write("Supported upload types:")
        st.code(", ".join(sorted(SUPPORTED_EXTENSIONS)))
        st.write("`.doc` is not supported directly. Convert it to `.docx`, `.pdf`, or `.txt`.")

    st.subheader("1) Upload Documents")
    uploads = st.file_uploader(
        "Upload one or more files",
        type=["txt", "md", "pdf", "docx"],
        accept_multiple_files=True,
    )
    if uploads:
        saved_paths = save_uploads(uploads)
        st.success(f"Saved {len(saved_paths)} files to docs/uploaded/")

    st.subheader("2) Build / Rebuild Index")
    col1, col2 = st.columns([1, 1])
    with col1:
        chunk_size = st.number_input("Chunk size", min_value=300, max_value=2000, value=900, step=50)
    with col2:
        overlap = st.number_input("Chunk overlap", min_value=0, max_value=500, value=150, step=10)

    if st.button("Build Index", type="primary"):
        with st.spinner("Building embeddings index..."):
            try:
                payload = build_index(
                    embedding_model=embedding_model,
                    docs_dir=Path("docs"),
                    out_path=index_path,
                    chunk_size=int(chunk_size),
                    overlap=int(overlap),
                    batch_size=64,
                )
                st.success(f"Index built: {len(payload['records'])} chunks")
            except Exception as exc:
                st.error(f"Index build failed: {exc}")

    if not index_path.exists():
        st.info("No index found yet. Upload docs and click Build Index.")
        st.stop()

    try:
        matrix, records, embed_model_from_index = load_index(index_path)
    except Exception as exc:
        st.error(f"Could not load index: {exc}")
        st.stop()

    st.subheader("3) Chat")
    st.caption(f"Loaded {len(records)} chunks from index using `{embed_model_from_index}`")

    if "history" not in st.session_state:
        st.session_state.history = []

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about your documents...")
    if question:
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, used_model = ask_once(
                        question=question,
                        model=model,
                        matrix=matrix,
                        records=records,
                        embedding_model=embed_model_from_index,
                        top_k=top_k,
                        temperature=float(temperature),
                    )
                    answer = f"{answer}\n\n_Answered with model: `{used_model}`_"
                except Exception as exc:
                    answer = f"Request failed: {exc}"
                st.markdown(answer)
        st.session_state.history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
