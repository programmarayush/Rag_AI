# RAG Chatbot (Gemini + Google Login)

This Streamlit app provides document-grounded Q&A with Gemini, protected by Google (Gmail) login.

## Features
- Google OAuth login gate
- Upload documents: `.txt`, `.md`, `.pdf`, `.docx`
- Build embedding index and retrieve relevant chunks
- Gemini grounded answers with source snippet IDs
- Adjustable `top_k` and `temperature` in UI

## Setup

```bash
cd rag_openai
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
cp .env.example .env
```

## Environment Variables

```bash
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=auto
GEMINI_EMBEDDING_MODEL=auto

GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
GOOGLE_REDIRECT_URI=https://your-app-url.streamlit.app
GOOGLE_STATE_SECRET=optional_long_random_string
GOOGLE_ALLOWED_DOMAIN=
GOOGLE_ALLOWED_EMAILS=
```

Notes:
- Keep `GEMINI_MODEL=auto` and `GEMINI_EMBEDDING_MODEL=auto` for compatibility fallback.
- `GOOGLE_ALLOWED_DOMAIN` and `GOOGLE_ALLOWED_EMAILS` are optional restrictions.

## Google OAuth Configuration

In Google Cloud Console (OAuth Client ID: Web application):
- Authorized JavaScript origins:
  - `https://<your-app>.streamlit.app`
- Authorized redirect URIs:
  - `https://<your-app>.streamlit.app`
  - `https://<your-app>.streamlit.app/`

Use the exact same base URL in `GOOGLE_REDIRECT_URI`.

## Run Locally

```bash
streamlit run app.py
```

## Streamlit Cloud Deployment

- Main file path: `app.py`
- Add secrets in TOML format:

```toml
GEMINI_API_KEY = "..."
GEMINI_MODEL = "auto"
GEMINI_EMBEDDING_MODEL = "auto"
GOOGLE_CLIENT_ID = "..."
GOOGLE_CLIENT_SECRET = "..."
GOOGLE_REDIRECT_URI = "https://<your-app>.streamlit.app"
GOOGLE_STATE_SECRET = "long-random-secret"
GOOGLE_ALLOWED_DOMAIN = ""
GOOGLE_ALLOWED_EMAILS = ""
```
