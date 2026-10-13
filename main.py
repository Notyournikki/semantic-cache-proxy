import os
import time
import json
import threading
import numpy as np
import faiss
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv(override=True)

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is missing from the .env file")

client = Groq(api_key=api_key)
app = FastAPI(title="Semantic Cache Proxy")

# --- Cache setup ---
model = SentenceTransformer("all-MiniLM-L6-v2")
THRESHOLD = 0.85
TTL_SECONDS = 24 * 60 * 60          # entries expire after 24 hours
DATA_DIR = "cache_data"
INDEX_FILE = os.path.join(DATA_DIR, "index.faiss")
STORE_FILE = os.path.join(DATA_DIR, "store.json")
os.makedirs(DATA_DIR, exist_ok=True)
lock = threading.Lock()             # prevents two requests editing the cache at once

index = faiss.IndexFlatIP(384)
store = []          # each entry: {"query", "answer", "tokens", "ts"}


def save_cache():
    faiss.write_index(index, INDEX_FILE)
    with open(STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(store, f)


def load_cache():
    global index, store
    if os.path.exists(INDEX_FILE) and os.path.exists(STORE_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(STORE_FILE, "r", encoding="utf-8") as f:
            store = json.load(f)
        print(f"Loaded {len(store)} cache entries from disk")


def remove_expired():
    """Rebuild the index without entries older than the TTL."""
    global index, store
    now = time.time()
    keep = [i for i, s in enumerate(store) if now - s["ts"] < TTL_SECONDS]
    if len(keep) == len(store):
        return
    new_index = faiss.IndexFlatIP(384)
    new_store = []
    for i in keep:
        new_index.add(index.reconstruct(i).reshape(1, -1))
        new_store.append(store[i])
    index, store = new_index, new_store
    save_cache()


load_cache()

# --- Stats setup ---
hit_latencies = []
miss_latencies = []
tokens_saved = 0
recent = []         # last 20 requests, used by the dashboard later


def embed(text: str) -> np.ndarray:
    v = model.encode([text], normalize_embeddings=True)
    return np.array(v, dtype="float32")


def call_llm(query: str):
    """Returns (answer, total_tokens_used)."""
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": query}],
    )
    answer = response.choices[0].message.content
    tokens = response.usage.total_tokens if response.usage else 0
    return answer, tokens


def log_recent(query, result, similarity, latency):
    recent.append({
        "query": query,
        "result": result,
        "similarity": similarity,
        "latency_ms": latency,
    })
    if len(recent) > 20:
        recent.pop(0)


class ChatRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {"message": "Semantic Cache Proxy is running"}


@app.post("/v1/chat")
def chat(request: ChatRequest):
    global tokens_saved
    start = time.time()
    try:
        vec = embed(request.query)

        # 1. Look for a similar question in the cache
        found = None
        with lock:
            remove_expired()
            if index.ntotal > 0:
                scores, ids = index.search(vec, 1)
                best_score = float(scores[0][0])
                if best_score >= THRESHOLD:
                    found = (store[int(ids[0][0])], best_score)

        if found:
            hit, best_score = found
            latency = round((time.time() - start) * 1000, 1)

            hit_latencies.append(latency)
            tokens_saved += hit["tokens"]
            log_recent(request.query, "HIT", round(best_score, 3), latency)

            return {
                "query": request.query,
                "matched_query": hit["query"],
                "answer": hit["answer"],
                "cache": "HIT",
                "similarity": round(best_score, 3),
                "latency_ms": latency,
            }

        # 2. Cache miss: ask the LLM, then remember the result
        answer, tokens = call_llm(request.query)
        
        with lock:
            index.add(vec)
            store.append({
                "query": request.query, 
                "answer": answer, 
                "tokens": tokens, 
                "ts": time.time()
            })
            save_cache()

        latency = round((time.time() - start) * 1000, 1)

        miss_latencies.append(latency)
        log_recent(request.query, "MISS", None, latency)

        return {
            "query": request.query,
            "answer": answer,
            "cache": "MISS",
            "latency_ms": latency,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def stats():
    hits = len(hit_latencies)
    misses = len(miss_latencies)
    total = hits + misses
    avg_hit = sum(hit_latencies) / hits if hits else 0
    avg_miss = sum(miss_latencies) / misses if misses else 0

    return {
        "total_requests": total,
        "hits": hits,
        "misses": misses,
        "hit_rate_percent": round(hits / total * 100, 1) if total else 0,
        "avg_hit_latency_ms": round(avg_hit, 1),
        "avg_miss_latency_ms": round(avg_miss, 1),
        "estimated_time_saved_ms": round((avg_miss - avg_hit) * hits, 1) if hits and misses else 0,
        "tokens_saved": tokens_saved,
        "cache_entries": len(store),
        "threshold": THRESHOLD,
        "recent_requests": recent,
    }


@app.get("/cache")
def view_cache():
    return {"entries": len(store), "queries": [s["query"] for s in store]}


@app.post("/reset")
def reset_cache():
    global tokens_saved
    with lock:
        index.reset()
        store.clear()
        save_cache()
    hit_latencies.clear()
    miss_latencies.clear()
    recent.clear()
    tokens_saved = 0
    return {"message": "Cache and stats cleared"}