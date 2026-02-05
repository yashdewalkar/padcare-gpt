# PadCare GPT (Public-Info RAG Assistant)

PadCare GPT is a Retrieval-Augmented Generation (RAG) assistant that answers questions about PadCare Labs using **only publicly available information** (website, public articles, public social pages). It is designed like an internal tool for fast, trustworthy understanding and content generation.

## How it works
1. **Data collection**: Fetches public webpages listed in `sources.yml`.
2. **Cleaning & structuring**: Removes common noise (nav/footers/scripts), keeps meaningful text, stores source metadata.
3. **Chunking & embeddings**: Splits text into small overlapping chunks and generates embeddings using SentenceTransformers.
4. **Vector database**: Stores embeddings in a local FAISS index.
5. **RAG pipeline** (query time): Embeds the user question → retrieves top-k relevant chunks → sends chunks + question to the LLM.
6. **Strict grounding (anti-hallucination)**: The LLM is instructed to answer only from the retrieved context. If missing, it responds:
   > "This information is not available in the provided sources."

## Sources used
All sources are public and listed in `sources.yml`. Example sources:
- Official PadCare website
- PadCare public LinkedIn company page
- Public news/press/blog coverage (e.g., Forbes India, WEF, etc.)

## Tech stack (free / open-source)
- LLM: **Mistral** running locally via **Ollama**
- Embeddings: **SentenceTransformers** (`all-MiniLM-L6-v2`)
- Vector DB: **FAISS**
- UI: **Streamlit**
- Language: Python
## Limitations
### 
-Responses may have higher latency since the LLM runs locally on CPU without hardware acceleration.

-The model is not fine-tuned on PadCare-specific data and depends entirely on retrieval quality.

-There is no formal evaluation framework or automated accuracy metrics.
## System Architecture
<img src="assets/rag_architecture.png" width="800"/>

## Streamlit UI
<img src="assets/Padcare_gpt.png" width="800"/> 

## Run locally (Windows)
### 1) Install Ollama + pull Mistral
```bash
ollama pull mistral