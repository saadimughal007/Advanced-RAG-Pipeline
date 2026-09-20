import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing from the .env file.")


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ─────────────────────────────────────────────
# Embedding Model
# ─────────────────────────────────────────────

embeddings = HuggingFaceEndpointEmbeddings(
    model=EMBEDDING_MODEL,
    huggingfacehub_api_token=HF_TOKEN,
)


# ─────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────

def create_embedding(text: str) -> np.ndarray:
    """Convert text into an embedding vector."""
    return np.array(embeddings.embed_query(text))


def cosine_similarity(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> float:
    """Calculate cosine similarity between two vectors."""

    denominator = (
        np.linalg.norm(vector_a) *
        np.linalg.norm(vector_b)
    )

    if denominator == 0:
        raise ValueError("Cannot calculate similarity for a zero vector.")

    return float(np.dot(vector_a, vector_b) / denominator)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:

    query = "Python programming seekhna aasan hai."

    related_text = (
        "Python code likhna bohot simple aur fast hai."
    )

    unrelated_text = (
        "Aaj ka mausam bohot thanda aur baarish wala hai."
    )

    # Create embeddings
    query_vector = create_embedding(query)
    related_vector = create_embedding(related_text)
    unrelated_vector = create_embedding(unrelated_text)

    # Calculate similarity
    related_score = cosine_similarity(
        query_vector,
        related_vector,
    )

    unrelated_score = cosine_similarity(
        query_vector,
        unrelated_vector,
    )

    # Display results
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"Vector Dimensions: {len(query_vector)}")

    print("\nSimilarity Results")
    print("-" * 40)

    print(f"Query vs Related Text:   {related_score:.4f}")
    print(f"Query vs Unrelated Text: {unrelated_score:.4f}")


if __name__ == "__main__":
    main()
    
