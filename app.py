import os
import base64
import hmac
import json
import secrets
import time
import urllib.parse
import urllib.request
from pathlib import Path

import streamlit as st
import google.generativeai as genai
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return False

from chat import ask_once, load_index
from ingest import SUPPORTED_EXTENSIONS, build_index

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"


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


def _pick_first(value):
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded.encode("utf-8"))


def _google_auth_config() -> dict:
    return {
        "client_id": os.getenv("GOOGLE_CLIENT_ID", "").strip(),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET", "").strip(),
        "redirect_uri": os.getenv("GOOGLE_REDIRECT_URI", "").strip(),
        "state_secret": os.getenv("GOOGLE_STATE_SECRET", "").strip(),
        "allowed_domain": os.getenv("GOOGLE_ALLOWED_DOMAIN", "").strip().lower(),
        "allowed_emails": {
            e.strip().lower()
            for e in os.getenv("GOOGLE_ALLOWED_EMAILS", "").split(",")
            if e.strip()
        },
    }


def _state_signing_key(cfg: dict) -> str:
    return cfg["state_secret"] or cfg["client_secret"]


def _create_state_token(cfg: dict, ttl_seconds: int = 600) -> str:
    payload = {
        "nonce": secrets.token_urlsafe(16),
        "iat": int(time.time()),
        "exp": int(time.time()) + ttl_seconds,
    }
    payload_json = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    payload_b64 = _b64url(payload_json)
    sig = hmac.new(
        _state_signing_key(cfg).encode("utf-8"),
        payload_b64.encode("utf-8"),
        digestmod="sha256",
    ).digest()
    return f"{payload_b64}.{_b64url(sig)}"


def _verify_state_token(cfg: dict, token: str) -> bool:
    try:
        payload_b64, sig_b64 = token.split(".", 1)
        expected_sig = hmac.new(
            _state_signing_key(cfg).encode("utf-8"),
            payload_b64.encode("utf-8"),
            digestmod="sha256",
        ).digest()
        got_sig = _b64url_decode(sig_b64)
        if not hmac.compare_digest(expected_sig, got_sig):
            return False
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
        now = int(time.time())
        return now <= int(payload.get("exp", 0))
    except Exception:
        return False


def _build_auth_url(client_id: str, redirect_uri: str, state: str) -> str:
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "prompt": "select_account",
    }
    return f"{AUTH_URL}?{urllib.parse.urlencode(params)}"


def _post_form(url: str, form_data: dict, headers: dict | None = None) -> dict:
    data = urllib.parse.urlencode(form_data).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(url: str, headers: dict | None = None) -> dict:
    req = urllib.request.Request(url, method="GET")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _exchange_code_for_user(code: str, client_id: str, client_secret: str, redirect_uri: str) -> dict:
    token_payload = {
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }
    token = _post_form(TOKEN_URL, token_payload)
    access_token = token.get("access_token")
    if not access_token:
        raise ValueError("Google token exchange failed: no access token.")
    return _get_json(USERINFO_URL, headers={"Authorization": f"Bearer {access_token}"})


def _render_google_login() -> bool:
    cfg = _google_auth_config()
    if not cfg["client_id"] or not cfg["client_secret"] or not cfg["redirect_uri"]:
        st.error(
            "Google OAuth is not configured. Set GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, and GOOGLE_REDIRECT_URI in secrets/env."
        )
        st.stop()

    params = st.query_params
    code = _pick_first(params.get("code"))
    state = _pick_first(params.get("state"))

    if "oauth_state" not in st.session_state:
        st.session_state.oauth_state = _create_state_token(cfg)

    if code:
        valid_state = bool(state) and (
            state == st.session_state.get("oauth_state") or _verify_state_token(cfg, state)
        )
        if not valid_state:
            st.warning("Login session expired. Please sign in again.")
            st.query_params.clear()
            st.session_state.oauth_state = _create_state_token(cfg)
            st.rerun()
        try:
            user = _exchange_code_for_user(
                code=code,
                client_id=cfg["client_id"],
                client_secret=cfg["client_secret"],
                redirect_uri=cfg["redirect_uri"],
            )
        except Exception as exc:
            st.error(f"Google login failed: {exc}")
            st.stop()

        email = str(user.get("email", "")).lower()
        domain = email.split("@")[-1] if "@" in email else ""
        allowed_domain = cfg["allowed_domain"]
        allowed_emails = cfg["allowed_emails"]
        if allowed_domain and domain != allowed_domain:
            st.error("Access denied: your domain is not allowed.")
            st.stop()
        if allowed_emails and email not in allowed_emails:
            st.error("Access denied: your email is not allowlisted.")
            st.stop()

        st.session_state.authenticated = True
        st.session_state.user = {
            "name": user.get("name") or email,
            "email": email,
            "picture": user.get("picture", ""),
        }
        st.query_params.clear()
        st.rerun()

    st.title("Login Required")
    st.caption("Sign in with your Google account to access the RAG application.")
    auth_url = _build_auth_url(
        cfg["client_id"],
        cfg["redirect_uri"],
        st.session_state["oauth_state"],
    )
    st.link_button("Sign in with Google", auth_url, use_container_width=True)
    return False


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="RAG Chatbot", page_icon="📄", layout="wide")

    if not st.session_state.get("authenticated", False):
        if not _render_google_login():
            return

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
        user = st.session_state.get("user", {})
        st.caption(f"Signed in as: `{user.get('email', 'unknown')}`")
        if st.button("Logout"):
            for k in ["authenticated", "user", "history", "oauth_state"]:
                st.session_state.pop(k, None)
            st.query_params.clear()
            st.rerun()
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
