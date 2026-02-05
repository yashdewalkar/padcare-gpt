import os
import re
import json
import yaml
import time
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROC_DIR = os.path.join(DATA_DIR, "processed")
INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_WORDS = 350       # approx 500-600 chars
CHUNK_OVERLAP_WORDS = 60 # small overlap for context

USER_AGENT = "PadCareGPT/1.0 (public-info-rag; contact: none)"

@dataclass
class DocChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]

def ensure_dirs():
    for d in [DATA_DIR, RAW_DIR, PROC_DIR, INDEX_DIR]:
        os.makedirs(d, exist_ok=True)

def stable_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def fetch_url(url: str, timeout: int = 30) -> str:
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text

def html_to_text(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")

    # Remove common noisy elements
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    # Prefer article/main content if present
    main = soup.find("main")
    if main is None:
        main = soup.find("article")
    if main is None:
        main = soup.body if soup.body else soup

    # Title
    title = soup.title.get_text(" ", strip=True) if soup.title else "Untitled"

    text = main.get_text("\n", strip=True)
    text = normalize_text(text)
    return title, text

def normalize_text(text: str) -> str:
    # Collapse excessive whitespace, remove very short noisy lines
    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if len(line) < 30:
            continue
        # drop cookie-banner-ish lines
        if re.search(r"(cookie|privacy policy|accept all|subscribe|sign up)", line, re.I):
            continue
        lines.append(line)
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned

def structure_by_topic(text: str) -> List[Tuple[str, str]]:
    """
    Very lightweight structuring:
    - Split by blank lines and treat each paragraph as a 'section'
    - Later you can enhance with heading detection
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sections = []
    for i, p in enumerate(paras):
        sections.append((f"section_{i+1}", p))
    return sections

def chunk_words(words: List[str], chunk_size: int, overlap: int) -> List[List[str]]:
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(words[start:end])
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks

def chunk_section(section_text: str) -> List[str]:
    words = section_text.split()
    word_chunks = chunk_words(words, CHUNK_WORDS, CHUNK_OVERLAP_WORDS)
    return [" ".join(wc).strip() for wc in word_chunks if len(wc) > 0]

def load_sources(path: str = "sources.yml") -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return obj["sources"]

def save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 32) -> np.ndarray:
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)

def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine sim if vectors normalized
    index.add(vectors)
    return index

def main():
    ensure_dirs()
    sources = load_sources()

    # 1) collecting and cleaning of data 
    raw_docs = []
    for s in tqdm(sources, desc="Fetching sources"):
        url = s["url"]
        name = s["name"]
        try:
            html = fetch_url(url)
            title, text = html_to_text(html)

            raw_path = os.path.join(RAW_DIR, f"{stable_id(url)}.json")
            save_json(raw_path, {"url": url, "name": name, "title": title, "text": text})

            raw_docs.append({"url": url, "name": name, "title": title, "text": text})
            time.sleep(0.6)  # be polite
        except Exception as e:
            print(f"[WARN] Failed: {url} | {e}")

    # 2) Structure + chunk
    chunks: List[DocChunk] = []
    for doc in tqdm(raw_docs, desc="Structuring + chunking"):
        sections = structure_by_topic(doc["text"])
        for section_name, section_text in sections:
            for j, chunk_text in enumerate(chunk_section(section_text)):
                meta = {
                    "source_name": doc["name"],
                    "source_url": doc["url"],
                    "doc_title": doc["title"],
                    "section": section_name,
                    "chunk_no": j + 1,
                }
                cid = stable_id(doc["url"] + "|" + section_name + "|" + str(j))
                chunks.append(DocChunk(chunk_id=cid, text=chunk_text, metadata=meta))

    if not chunks:
        raise RuntimeError("No chunks produced. Check your sources.yml or network access.")

    # Saveing processed chunks
    proc_path = os.path.join(PROC_DIR, "chunks.jsonl")
    with open(proc_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"id": c.chunk_id, "text": c.text, "metadata": c.metadata}, ensure_ascii=False) + "\n")
    print(f"[OK] Saved chunks: {proc_path} ({len(chunks)} chunks)")

    # 3) Embeding + index
    model = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [c.text for c in chunks]
    vectors = embed_texts(model, texts)

    index = build_faiss_index(vectors)
    faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))

    # Store metadata aligned by vector row
    meta_path = os.path.join(INDEX_DIR, "meta.json")
    save_json(meta_path, {
        "embed_model": EMBED_MODEL_NAME,
        "chunk_count": len(chunks),
        "items": [
            {"id": c.chunk_id, "text": c.text, "metadata": c.metadata}
            for c in chunks
        ]
    })

    print(f"[OK] FAISS index saved to: {INDEX_DIR}")

if __name__ == "__main__":
    main()