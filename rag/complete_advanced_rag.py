import warnings
warnings.filterwarnings("ignore", message="Direct use of automatic function calling.*")

import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from sentence_transformers import CrossEncoder

# ============================================================
# SETUP
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing from the .env file.")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing from the .env file.")


# Embeddings
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=HF_TOKEN,
)

# Vector Store
vector_store = Chroma(
    collection_name="knowledge_base",
    embedding_function=embeddings,
    persist_directory=str(BASE_DIR / "chroma_db"),
)

# MMR Retriever
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 10,
    },
)

# LLM
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GOOGLE_API_KEY,
    thinking_level="low",
    max_output_tokens=500,
)

# Reranker
reranker = CrossEncoder("BAAI/bge-reranker-base")


# ============================================================
# STAGE 1: QUERY REWRITING
# ============================================================

rewrite_prompt = ChatPromptTemplate.from_template(
    """
    Rewrite the user's question into ONE clear search query
    optimized for retrieving relevant documents.

    Keep the original meaning.
    Remove unnecessary conversational words.
    Focus on important technical concepts and keywords.

    Return only the rewritten search query. No extra text.

    User question:
    {question}
    """
)


def extract_text(content):
    """Convert Gemini response to plain text"""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "\n".join(text_parts)

    return str(content)


def stage1_rewrite_query(question: str) -> str:
    """
    Stage 1: Rewrite user's question into optimized search query
    """
    print("\n[STAGE 1] Query Rewriting...")

    messages = rewrite_prompt.invoke({"question": question})
    response = model.invoke(messages)
    rewritten = extract_text(response.content).strip()

    print(f"  Original: {question}")
    print(f"  Rewritten: {rewritten}")

    return rewritten


# ============================================================
# STAGE 2: MULTI-QUERY RETRIEVAL + DEDUPLICATION
# ============================================================

multi_query_prompt = ChatPromptTemplate.from_template(
    """
    Generate 3 different search queries for the user's question.

    The queries should approach the question from different angles.
    Make them diverse but relevant.

    Return only the 3 queries, one per line. No numbering or extra text.

    User question:
    {question}
    """
)


def generate_multiple_queries(question: str) -> list:
    """Generate 3 different search queries"""
    messages = multi_query_prompt.invoke({"question": question})
    response = model.invoke(messages)
    content = extract_text(response.content)

    queries = [
        line.strip()
        for line in content.splitlines()
        if line.strip()
    ]

    return queries[:3]


def remove_duplicate_documents(documents: list) -> list:
    """Remove exact duplicate documents based on content"""
    unique_documents = []
    seen = set()

    for document in documents:
        content = document.page_content

        if content not in seen:
            seen.add(content)
            unique_documents.append(document)

    return unique_documents


def stage2_multi_query_retrieve(question: str) -> tuple:
    """
    Stage 2: Generate multiple queries and retrieve documents
    Returns: (generated_queries, unique_documents)
    """
    print("\n[STAGE 2] Multi-Query Retrieval + Deduplication...")

    # Generate 3 different queries
    queries = generate_multiple_queries(question)

    print(f"  Generated {len(queries)} queries:")
    for i, q in enumerate(queries, 1):
        print(f"    {i}. {q}")

    # Retrieve documents using each query
    all_documents = []
    for query in queries:
        docs = retriever.invoke(query)
        all_documents.extend(docs)

    print(f"  Retrieved {len(all_documents)} total documents")

    # Remove duplicates
    unique_documents = remove_duplicate_documents(all_documents)

    print(f"  After deduplication: {len(unique_documents)} unique documents")

    return queries, unique_documents


# ============================================================
# STAGE 3: RERANKING
# ============================================================

def stage3_rerank_documents(
    question: str,
    documents: list,
    top_k: int = 3,
) -> list:
    """
    Stage 3: Rerank documents using CrossEncoder
    Returns: List of (score, document) tuples for top_k documents
    """
    print(f"\n[STAGE 3] Reranking (selecting top {top_k})...")

    # Create (question, document_content) pairs
    pairs = [
        (question, document.page_content)
        for document in documents
    ]

    # Get relevance scores
    scores = reranker.predict(pairs)

    # Sort by score (highest first)
    ranked_documents = sorted(
        zip(scores, documents),
        key=lambda item: item[0],
        reverse=True,
    )

    print(f"  Top {top_k} documents by relevance:")
    for i, (score, doc) in enumerate(ranked_documents[:top_k], 1):
        source = doc.metadata.get("source", "Unknown")
        print(f"    {i}. Score: {score:.4f} | Source: {source}")

    return ranked_documents[:top_k]


# ============================================================
# STAGE 4: CONTEXT COMPRESSION
# ============================================================

compress_prompt = ChatPromptTemplate.from_template(
    """
    You are a document compression expert.

    Extract only the relevant information from the given text
    that answers the user's question.

    Remove unnecessary explanations, examples, and filler.
    Keep only the essential parts that directly address the question.

    Question: {question}

    Text to compress:
    {text}

    Compressed text (extract only relevant parts):
    """
)


def compress_single_document(question: str, document_text: str) -> str:
    """Compress a single document using LLM"""
    messages = compress_prompt.invoke({
        "question": question,
        "text": document_text,
    })

    response = model.invoke(messages)
    compressed = extract_text(response.content).strip()

    return compressed


def stage4_compress_documents(
    question: str,
    ranked_documents: list,
) -> list:
    """
    Stage 4: Compress reranked documents
    Returns: List of compressed documents (with updated page_content)
    """
    print("\n[STAGE 4] Context Compression...")

    compressed_documents = []

    for score, document in ranked_documents:
        # Compress the document
        compressed_text = compress_single_document(
            question,
            document.page_content,
        )

        # Create new document with compressed content
        from langchain_core.documents import Document

        compressed_doc = Document(
            page_content=compressed_text,
            metadata=document.metadata,
        )

        compressed_documents.append(compressed_doc)

    print(f"  Compressed {len(compressed_documents)} documents")

    return compressed_documents


# ============================================================
# STAGE 5: FORMAT CONTEXT + FINAL LLM ANSWER
# ============================================================

final_prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful AI assistant.

    Answer the user's question using ONLY the provided context.

    If the answer is not present in the context,
    clearly state that you do not have enough information.

    Provide a clear, concise, and well-structured answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)


def format_documents_as_context(documents: list) -> str:
    """Format compressed documents into context string"""
    context_parts = []

    for i, document in enumerate(documents, 1):
        source = document.metadata.get("source", "Unknown")
        content = document.page_content

        context_parts.append(f"[Source {i}: {source}]\n{content}")

    return "\n\n".join(context_parts)


def stage5_generate_answer(
    question: str,
    compressed_documents: list,
) -> dict:
    """
    Stage 5: Generate final answer using compressed documents
    Returns: {"answer": str, "sources": list}
    """
    print("\n[STAGE 5] Generating Final Answer...")

    # Format context
    context = format_documents_as_context(compressed_documents)

    # Generate answer
    messages = final_prompt.invoke({
        "context": context,
        "question": question,
    })

    response = model.invoke(messages)
    answer = extract_text(response.content).strip()

    # Collect sources
    sources = [
        doc.metadata.get("source", "Unknown")
        for doc in compressed_documents
    ]

    print(f"  Answer generated from {len(sources)} sources")

    return {
        "answer": answer,
        "sources": sources,
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_advanced_rag(question: str) -> dict:
    """
    Run complete advanced RAG pipeline:
    
    1. Query Rewriting - Optimize the question
    2. Multi-Query Retrieval - Search with multiple queries
    3. Deduplication - Remove duplicate documents
    4. Reranking - Sort by relevance (CrossEncoder)
    5. Context Compression - Extract only relevant info
    6. Final Answer - Generate answer with LLM
    
    Returns: {"answer": str, "sources": list, "pipeline_info": dict}
    """
    print("\n" + "=" * 60)
    print("COMPLETE ADVANCED RAG PIPELINE")
    print("=" * 60)

    print(f"\nQuestion: {question}")

    # Stage 1: Rewrite query
    rewritten_query = stage1_rewrite_query(question)

    # Stage 2: Multi-query retrieval + dedup
    queries, unique_documents = stage2_multi_query_retrieve(question)

    # Stage 3: Reranking
    ranked_documents = stage3_rerank_documents(
        question,
        unique_documents,
        top_k=3,
    )

    # Stage 4: Context compression
    compressed_documents = stage4_compress_documents(
        question,
        ranked_documents,
    )

    # Stage 5: Generate final answer
    result = stage5_generate_answer(question, compressed_documents)

    # Add pipeline info
    result["pipeline_info"] = {
        "original_question": question,
        "rewritten_query": rewritten_query,
        "generated_queries": queries,
        "total_retrieved": len(unique_documents),
        "reranked_top_k": len(ranked_documents),
        "compressed_documents": len(compressed_documents),
    }

    return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    question = input("Ask a question: ")

    result = run_advanced_rag(question)

    # Display results
    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources Used:")
    for source in result["sources"]:
        print(f"  - {source}")

    print("\nPipeline Info:")
    for key, value in result["pipeline_info"].items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    