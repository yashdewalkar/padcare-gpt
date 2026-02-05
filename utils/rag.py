import os
import json
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests

INDEX_DIR = os.path.join("data", "faiss_index")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

def load_index() -> Tuple[faiss.Index, Dict[str, Any]]:
    index = faiss.read_index(os.path.join(INDEX_DIR, "index.faiss"))
    with open(os.path.join(INDEX_DIR, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta

def embed_query(model: SentenceTransformer, q: str) -> np.ndarray:
    v = model.encode([q], normalize_embeddings=True)
    return np.asarray(v, dtype=np.float32)

def search(index: faiss.Index, q_vec: np.ndarray, k: int = 5) -> List[int]:
    scores, ids = index.search(q_vec, k)
    return ids[0].tolist()

def build_prompt(retrieved: List[Dict[str, Any]], question: str, intent: str) -> str:
    # strict instruction as per the requirement in the assignment
    system = (
        "You are PadCare GPT.\n"
        "Answer ONLY using the provided context.\n"
        "If information is missing, respond with:\n"
        "\"This information is not available in the provided sources.\"\n"
        "Do not use outside knowledge.\n"
        "Always include citations in the format [source_name - source_url].\n"
    )

    format_instructions = {
        "qa": "Write a direct, factual answer grounded in the context.",
        "simple": "Explain in simple words (as if to a non-technical person). Keep it short.",
        "investor_pitch": "Write a short investor pitch (6–8 lines) grounded in the context.",
        "linkedin_post": "Write an engaging LinkedIn post (120–180 words) grounded in the context. Add 3–6 hashtags.",
        "bullets": "Summarize PadCare in exactly 5 bullet points grounded in the context.",
    }.get(intent, "Write a grounded answer.")

    context_lines = []
    for i, item in enumerate(retrieved, 1):
        m = item["metadata"]
        context_lines.append(
            f"({i}) {item['text']}\n"
            f"SOURCE: {m['source_name']} | {m['source_url']}\n"
        )
    context = "\n".join(context_lines).strip()

    prompt = (
        f"{system}\n"
        f"Task: {format_instructions}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    return prompt

def ollama_generate(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2
        }
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("response", "").strip()

def answer(question: str, intent: str = "qa", k: int = 5) -> Dict[str, Any]:
    index, meta = load_index()
    embed_model = SentenceTransformer(meta["embed_model"])

    q_vec = embed_query(embed_model, question)
    idxs = search(index, q_vec, k=k)

    items = meta["items"]
    retrieved = [items[i] for i in idxs if 0 <= i < len(items)]

    prompt = build_prompt(retrieved, question, intent)
    response = ollama_generate(prompt)

    return {
        "answer": response,
        "retrieved": retrieved
    }