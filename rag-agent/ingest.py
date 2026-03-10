"""
ingest.py
Run this once to chunk your podcast transcripts, embed them, and upsert into Pinecone.

Install deps:
    pip install pinecone sentence-transformers python-dotenv
"""

import os
import re
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME       = "yfiob-rag-agent"
EMBED_MODEL      = "avsolatorio/GIST-large-Embedding-v0"  # 1024-dim, higher quality
CHUNK_SIZE       = 1000
CHUNK_OVERLAP    = 200
DATA_FOLDER      = "data/"

VALID_INDUSTRY_SECTORS = {
    "Architecture and Engineering",
    "Agriculture and Natural Resources",
    "Marketing, Sales, and Service",
    "Building, Trades, and Construction",
    "Energy, Environment, Utilities",
    "Fashion and Interior Design",
    "Manufacturing and Product Development",
    "Education, Child Development, Family Services",
    "Public and Government Services",
    "Finance and Business",
    "Arts, Media, and Entertainment",
    "Information and Computer Technologies",
    "Hospitality, Tourism, Recreation",
    "Health Services, Sciences, Medical Technology",
}


def load_transcripts(folder: str) -> list[dict]:
    transcripts = []
    for fname in os.listdir(folder):
        if not fname.endswith(".txt"):
            continue
        ep_id = fname.replace(".txt", "")
        with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
            content = f.read()

        # Extract interviewee name
        m = re.search(r'Interviewee:\s*([A-Za-z]+\s+[A-Za-z]+)', content)
        interviewee = m.group(1).strip() if m else "Unknown"

        # Extract industry sectors (only valid ones)
        m = re.search(r'Industry Sectors\s*:\s*([^\n]+?)(?=\s*Takeaways:|#|$)', content)
        industry_sectors = []
        if m:
            for sector in VALID_INDUSTRY_SECTORS:
                if sector in m.group(1):
                    industry_sectors.append(sector)

        # Extract source
        m = re.search(r'Source\s*:\s*([^\n]+)', content)
        source = m.group(1).strip() if m else "Unknown"

        transcripts.append({
            "id":               ep_id,
            "interviewee":      interviewee,
            "industry_sectors": industry_sectors,
            "source":           source,
            "text":             content,
        })

    print(f"Loaded {len(transcripts)} transcripts from '{folder}'")
    return transcripts


def chunk_text(text: str) -> list[str]:
    """Recursive character-aware chunking with overlap."""
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build_vectors(transcripts: list[dict], model: SentenceTransformer) -> list[dict]:
    vectors = []
    for ep in transcripts:
        chunks = chunk_text(ep["text"])
        embeddings = model.encode(chunks, show_progress_bar=False)
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            vectors.append({
                "id":     f"{ep['id']}_chunk{i}",
                "values": emb.tolist(),
                "metadata": {
                    "file_name":        ep["id"],
                    "chunk_id":         i,
                    "Interviewee":      ep["interviewee"],
                    "Industry Sectors": ep["industry_sectors"],
                    "Source":           ep["source"],
                    "content":          chunk,
                },
            })
    return vectors


def main():
    # 1. Init Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME not in existing:
        print(f"Creating index '{INDEX_NAME}' ...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1024,   # matches GIST-large-Embedding-v0
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    else:
        print(f"Index '{INDEX_NAME}' already exists, skipping creation.")

    index = pc.Index(INDEX_NAME)

    # 2. Load transcripts
    transcripts = load_transcripts(DATA_FOLDER)

    # 3. Embed
    print("Loading embedding model (first run will download it) ...")
    model = SentenceTransformer(EMBED_MODEL)

    print("Chunking & embedding ...")
    vectors = build_vectors(transcripts, model)
    print(f"  → {len(vectors)} chunks generated")

    # 4. Upsert in batches of 50
    for i in range(0, len(vectors), 50):
        index.upsert(vectors=vectors[i:i + 50])
        print(f"  Upserted batch {i // 50 + 1}")

    print("✅ Done!")
    print(index.describe_index_stats())


if __name__ == "__main__":
    main()