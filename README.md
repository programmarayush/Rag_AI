# RAG Chatbot (Gemini + Local Documents)

This is a separate project that answers questions from your own documents.
It uses:
- Gemini embeddings for retrieval
- cosine similarity for top-k chunk search
- Gemini chat model for grounded answers
- Streamlit UI for upload + chat

## 1. Setup

```bash
cd rag_openai
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
cp .env.example .env
```

Set your key in `.env`:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=auto
GEMINI_EMBEDDING_MODEL=auto
```

## 2. Add Documents

Put your files in `docs/`:
- `.txt`
- `.md`
- `.pdf`
- `.docx`

`.doc` is not supported directly. Convert `.doc` to `.docx`, `.pdf`, or `.txt`.

Example file already exists at `docs/sample_policy.txt`.

## 3. Build Index

```bash
python3 ingest.py --docs docs --out index/index.json
```

## 4. Ask Questions (One-shot)

```bash
python3 chat.py --index index/index.json --question "What is the peak demand reduction target?"
```

## 5. Interactive Chat

```bash
python3 chat.py --index index/index.json
```

Type `exit` to quit.

## 6. Streamlit Web UI

```bash
streamlit run app.py
```

In the UI:
1. Upload docs (`.txt`, `.md`, `.pdf`, `.docx`)
2. Click **Build Index**
3. Ask questions in chat

## Notes

- Answers are grounded in retrieved snippets only.
- If not found, it says it could not find it in the provided docs.
- Re-run `ingest.py` whenever documents change.
