import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from docx import Document
import google.generativeai as genai
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


def read_text_from_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        try:
            reader = PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)
        except Exception as exc:
            print(f"Warning: failed to read PDF {path}: {exc}")
            return ""
    if suffix == ".docx":
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text)
    return ""


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def batched(items: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def collect_records_from_docs(
    docs_dir: Path, chunk_size: int, overlap: int
) -> tuple[list[Path], list[dict]]:
    files = sorted(
        [
            p
            for p in docs_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )
    if not files:
        exts = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"No supported files ({exts}) found in {docs_dir}")

    records: list[dict] = []
    for path in files:
        raw = read_text_from_file(path)
        if not raw.strip():
            continue
        chunks = chunk_text(raw, chunk_size=chunk_size, overlap=overlap)
        for i, chunk in enumerate(chunks):
            records.append(
                {
                    "id": f"{path.name}#chunk-{i}",
                    "source": str(path),
                    "chunk_index": i,
                    "text": chunk,
                }
            )
    return files, records


def discover_embedding_models() -> list[str]:
    try:
        models = genai.list_models()
    except Exception:
        return []

    names: list[str] = []
    for m in models:
        methods = set(getattr(m, "supported_generation_methods", []) or [])
        if "embedContent" in methods and getattr(m, "name", None):
            names.append(str(m.name))
    return names


def build_candidate_models(configured_model: str) -> list[str]:
    candidates: list[str] = []
    seen = set()

    def add(name: str) -> None:
        if name and name not in seen:
            candidates.append(name)
            seen.add(name)

    cfg = configured_model.strip()
    if cfg and cfg.lower() != "auto":
        add(cfg)
        if cfg.startswith("models/"):
            add(cfg.split("/", 1)[1])
        else:
            add(f"models/{cfg}")

    add("models/embedding-001")
    add("embedding-001")
    add("models/text-embedding-004")
    add("text-embedding-004")

    for name in discover_embedding_models():
        add(name)

    return candidates


def embed_with_fallback(text: str, configured_model: str) -> tuple[list[float], str]:
    candidate_models = build_candidate_models(configured_model)
    last_exc: Exception | None = None
    for model_name in candidate_models:
        try:
            resp = genai.embed_content(
                model=model_name,
                content=text,
                task_type="retrieval_document",
            )
            return resp["embedding"], model_name
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Embedding failed with all candidate models.")


def build_index(
    embedding_model: str,
    docs_dir: Path,
    out_path: Path,
    chunk_size: int = 900,
    overlap: int = 150,
    batch_size: int = 64,
) -> dict:
    files, records = collect_records_from_docs(docs_dir, chunk_size, overlap)
    if not records:
        raise ValueError("No content extracted from provided files.")

    print(f"Embedding {len(records)} chunks from {len(files)} files...")
    texts = [r["text"] for r in records]
    vectors: list[list[float]] = []
    selected_model = embedding_model
    for batch in batched(texts, batch_size):
        for text in batch:
            emb, used_model = embed_with_fallback(text, selected_model)
            selected_model = used_model
            vectors.append(emb)

    if len(vectors) != len(records):
        raise RuntimeError("Embedding size mismatch.")

    for i, vec in enumerate(vectors):
        records[i]["embedding"] = vec

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "embedding_model": selected_model,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "records": records,
    }
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Saved index: {out_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an embedding index from local docs.")
    parser.add_argument(
        "--docs", type=str, default="docs", help="Folder with .txt/.md/.pdf/.docx files."
    )
    parser.add_argument("--out", type=str, default="index/index.json", help="Output index file path.")
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing. Set it in rag_openai/.env.")
    genai.configure(api_key=api_key)
    embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "auto")

    docs_dir = Path(args.docs)
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs folder not found: {docs_dir}")

    build_index(
        embedding_model=embedding_model,
        docs_dir=docs_dir,
        out_path=Path(args.out),
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
