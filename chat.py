import argparse
import json
import os
from pathlib import Path

import numpy as np
import google.generativeai as genai
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return False


SYSTEM_PROMPT = """You are a document-grounded assistant.
Answer only from the provided context snippets.
If the answer is not present in the context, say: "I could not find that in the provided documents."
Always include a short "Sources" section listing snippet IDs you used."""


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms


def load_index(path: Path) -> tuple[np.ndarray, list[dict], str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Index not found: {path}. Run ingest.py first to create it."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["records"]
    embedding_model = payload["embedding_model"]
    matrix = np.array([r["embedding"] for r in records], dtype=np.float32)
    matrix = l2_normalize(matrix)
    return matrix, records, embedding_model


def retrieve_top_k(
    query: str,
    matrix: np.ndarray,
    records: list[dict],
    embedding_model: str,
    top_k: int,
) -> list[tuple[float, dict]]:
    q = genai.embed_content(
        model=embedding_model,
        content=query,
        task_type="retrieval_query",
    )["embedding"]
    qv = np.array(q, dtype=np.float32).reshape(1, -1)
    qv = l2_normalize(qv)

    scores = (matrix @ qv.T).squeeze(-1)
    idx = np.argsort(-scores)[:top_k]
    return [(float(scores[i]), records[int(i)]) for i in idx]


def build_context(retrieved: list[tuple[float, dict]]) -> str:
    blocks = []
    for score, rec in retrieved:
        blocks.append(
            f"[{rec['id']}] (score={score:.3f}, source={rec['source']})\n{rec['text']}"
        )
    return "\n\n".join(blocks)


def discover_generation_models() -> list[str]:
    try:
        models = genai.list_models()
    except Exception:
        return []

    names: list[str] = []
    for m in models:
        methods = set(getattr(m, "supported_generation_methods", []) or [])
        if "generateContent" in methods and getattr(m, "name", None):
            names.append(str(m.name))
    return names


def build_generation_candidates(configured_model: str) -> list[str]:
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

    add("models/gemini-2.0-flash")
    add("gemini-2.0-flash")
    add("models/gemini-1.5-flash")
    add("gemini-1.5-flash")

    for name in discover_generation_models():
        add(name)
    return candidates


def generate_with_fallback(
    prompt: str, configured_model: str, temperature: float
) -> tuple[str, str]:
    last_exc: Exception | None = None
    for model_name in build_generation_candidates(configured_model):
        try:
            response = genai.GenerativeModel(model_name=model_name).generate_content(
                prompt,
                generation_config={"temperature": temperature},
            )
            text = response.text or "I could not find that in the provided documents."
            return text, model_name
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("No Gemini model available for generateContent.")


def ask_once(
    question: str,
    model: str,
    matrix: np.ndarray,
    records: list[dict],
    embedding_model: str,
    top_k: int,
    temperature: float = 0.2,
) -> tuple[str, str]:
    retrieved = retrieve_top_k(
        query=question,
        matrix=matrix,
        records=records,
        embedding_model=embedding_model,
        top_k=top_k,
    )
    context = build_context(retrieved)
    user_message = f"Question: {question}\n\nContext:\n{context}"

    answer, used_model = generate_with_fallback(
        f"{SYSTEM_PROMPT}\n\n{user_message}",
        model,
        temperature,
    )
    return answer, used_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive RAG chat over your local documents.")
    parser.add_argument("--index", type=str, default="index/index.json")
    parser.add_argument("--model", type=str, default=None, help="Override GEMINI_MODEL")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--question", type=str, default=None, help="One-shot question")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing. Set it in rag_openai/.env.")
    genai.configure(api_key=api_key)
    model = args.model or os.getenv("GEMINI_MODEL", "auto")

    matrix, records, embedding_model = load_index(Path(args.index))
    print(
        f"Loaded {len(records)} chunks | embedding={embedding_model} | model={model}"
    )

    if args.question:
        answer, used_model = ask_once(
            args.question,
            model,
            matrix,
            records,
            embedding_model,
            args.top_k,
            args.temperature,
        )
        print(f"[model={used_model}]")
        print(answer)
        return

    print("RAG Chat (type 'exit' to quit)\n")
    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        answer, used_model = ask_once(
            q, model, matrix, records, embedding_model, args.top_k, args.temperature
        )
        print(f"[model={used_model}]")
        print(f"Assistant: {answer}\n")


if __name__ == "__main__":
    main()
