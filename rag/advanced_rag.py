import os
import time
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from sentence_transformers import CrossEncoder


# ============================================================
# SETUP
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is missing from the .env file."
    )


# ============================================================
# LOCAL EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# VECTOR STORE
# ============================================================

vector_store = Chroma(
    collection_name="knowledge_base",
    embedding_function=embeddings,
    persist_directory=str(BASE_DIR / "chroma_db"),
)


# ============================================================
# RETRIEVER
# ============================================================

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 8,
    },
)


# ============================================================
# GEMINI
# ============================================================

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GOOGLE_API_KEY,
    thinking_level="low",
    max_output_tokens=500,
)


# ============================================================
# RERANKER
# ============================================================

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)


# ============================================================
# QUERY GENERATION
# ONE LLM CALL
# ============================================================

query_prompt = ChatPromptTemplate.from_template(
    """
You are a search-query generation system.

The user will provide one question.

Generate exactly 3 search queries that can be used
to retrieve relevant documents from a technical knowledge base.

Requirements:

- Query 1: direct interpretation of the question.
- Query 2: different technical angle.
- Query 3: practical implementation angle.
- Keep all queries concise.
- Preserve the original meaning.
- Do not answer the question.
- Return ONLY 3 queries, one per line.
- Do not number them.

User question:
{question}
"""
)


def extract_text(content) -> str:

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


async def generate_queries(question: str) -> list:

    messages = query_prompt.invoke({
        "question": question
    })

    response = await model.ainvoke(messages)

    content = extract_text(
        response.content
    )

    queries = [
        line.strip()
        for line in content.splitlines()
        if line.strip()
    ]

    return queries[:3]


# ============================================================
# PARALLEL RETRIEVAL
# ============================================================

async def retrieve_single_query(query: str):

    return await asyncio.to_thread(
        retriever.invoke,
        query
    )


async def retrieve_documents(queries: list):

    results = await asyncio.gather(
        *[
            retrieve_single_query(query)
            for query in queries
        ]
    )

    all_documents = []

    for documents in results:
        all_documents.extend(documents)

    return all_documents


# ============================================================
# DEDUPLICATION
# ============================================================

def remove_duplicate_documents(
    documents: list,
) -> list:

    unique_documents = []
    seen = set()

    for document in documents:

        content = document.page_content

        if content not in seen:

            seen.add(content)
            unique_documents.append(document)

    return unique_documents


# ============================================================
# RERANKING
# ============================================================

def rerank_documents(
    question: str,
    documents: list,
    top_k: int = 3,
) -> list:

    pairs = [
        (
            question,
            document.page_content
        )
        for document in documents
    ]

    scores = reranker.predict(
        pairs,
        show_progress_bar=False,
    )

    ranked_documents = sorted(
        zip(scores, documents),
        key=lambda item: item[0],
        reverse=True,
    )

    return ranked_documents[:top_k]


# ============================================================
# CONTEXT COMPRESSION
# ============================================================

compression_prompt = ChatPromptTemplate.from_template(
    """
You are a context compression system.

Extract ONLY the information from the document
that is directly relevant to answering the question.

Remove:

- unrelated information
- unnecessary examples
- repetition
- filler

Do not add new information.

If the document contains no useful information
for the question, return an empty string.

Question:
{question}

Document:
{text}

Return only the compressed relevant content.
"""
)


async def compress_single_document(
    question: str,
    document: Document,
):

    messages = compression_prompt.invoke({
        "question": question,
        "text": document.page_content,
    })

    response = await model.ainvoke(messages)

    compressed_text = extract_text(
        response.content
    ).strip()

    return Document(
        page_content=compressed_text,
        metadata=document.metadata,
    )


async def compress_documents(
    question: str,
    ranked_documents: list,
):

    compressed_documents = await asyncio.gather(
        *[
            compress_single_document(
                question,
                document,
            )
            for score, document in ranked_documents
        ]
    )

    # Remove empty compression results
    compressed_documents = [
        document
        for document in compressed_documents
        if document.page_content.strip()
    ]

    return compressed_documents


# ============================================================
# FINAL ANSWER
# ============================================================

final_prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Answer the user's question using ONLY the
provided context.

If the answer is not present in the context,
clearly say that you do not have enough information.

Provide a clear, concise and technically accurate answer.

Context:
{context}

Question:
{question}

Answer:
"""
)


def format_context(
    documents: list,
) -> str:

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1,
    ):

        source = document.metadata.get(
            "source",
            "Unknown",
        )

        context_parts.append(
            f"[Source {index}: {source}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(context_parts)


async def generate_answer(
    question: str,
    documents: list,
):

    context = format_context(
        documents
    )

    messages = final_prompt.invoke({
        "context": context,
        "question": question,
    })

    response = await model.ainvoke(
        messages
    )

    return extract_text(
        response.content
    ).strip()


# ============================================================
# COMPLETE PIPELINE
# ============================================================

async def run_advanced_rag(
    question: str,
) -> dict:

    total_start = time.perf_counter()

    print("\n" + "=" * 60)
    print("OPTIMIZED ADVANCED RAG")
    print("=" * 60)

    print(f"\nQuestion: {question}")


    # --------------------------------------------------------
    # Stage 1: Query Generation
    # --------------------------------------------------------

    start = time.perf_counter()

    queries = await generate_queries(
        question
    )

    query_time = time.perf_counter() - start

    print("\n[STAGE 1] Query Generation")

    for index, query in enumerate(
        queries,
        start=1,
    ):
        print(
            f"  {index}. {query}"
        )

    print(
        f"  Time: {query_time:.2f}s"
    )


    # --------------------------------------------------------
    # Stage 2: Parallel Retrieval
    # --------------------------------------------------------

    start = time.perf_counter()

    all_documents = await retrieve_documents(
        queries
    )

    unique_documents = remove_duplicate_documents(
        all_documents
    )

    retrieval_time = time.perf_counter() - start

    print("\n[STAGE 2] Parallel Retrieval")

    print(
        f"  Retrieved: {len(all_documents)}"
    )

    print(
        f"  After deduplication: "
        f"{len(unique_documents)}"
    )

    print(
        f"  Time: {retrieval_time:.2f}s"
    )


    # --------------------------------------------------------
    # Stage 3: Reranking
    # --------------------------------------------------------

    start = time.perf_counter()

    ranked_documents = rerank_documents(
        question,
        unique_documents,
        top_k=3,
    )

    rerank_time = time.perf_counter() - start

    print("\n[STAGE 3] Reranking")

    for index, (
        score,
        document,
    ) in enumerate(
        ranked_documents,
        start=1,
    ):

        print(
            f"  {index}. "
            f"Score: {score:.4f}"
        )

    print(
        f"  Time: {rerank_time:.2f}s"
    )


    # --------------------------------------------------------
    # Stage 4: Parallel Compression
    # --------------------------------------------------------

    start = time.perf_counter()

    compressed_documents = await compress_documents(
        question,
        ranked_documents,
    )

    compression_time = (
        time.perf_counter() - start
    )

    print("\n[STAGE 4] Parallel Compression")

    print(
        f"  Compressed documents: "
        f"{len(compressed_documents)}"
    )

    print(
        f"  Time: {compression_time:.2f}s"
    )


    # --------------------------------------------------------
    # Stage 5: Final Answer
    # --------------------------------------------------------

    start = time.perf_counter()

    answer = await generate_answer(
        question,
        compressed_documents,
    )

    answer_time = time.perf_counter() - start

    print("\n[STAGE 5] Final Answer")

    print(
        f"  Time: {answer_time:.2f}s"
    )


    # --------------------------------------------------------
    # Total
    # --------------------------------------------------------

    total_time = (
        time.perf_counter()
        - total_start
    )

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print("\n" + answer)

    print("\nSources:")

    sources = []

    for document in compressed_documents:

        source = document.metadata.get(
            "source",
            "Unknown",
        )

        if source not in sources:
            sources.append(source)

            print(
                f"  - {source}"
            )


    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PERFORMANCE")
    print("=" * 60)

    print(
        f"Query Generation: "
        f"{query_time:.2f}s"
    )

    print(
        f"Retrieval: "
        f"{retrieval_time:.2f}s"
    )

    print(
        f"Reranking: "
        f"{rerank_time:.2f}s"
    )

    print(
        f"Compression: "
        f"{compression_time:.2f}s"
    )

    print(
        f"Final Answer: "
        f"{answer_time:.2f}s"
    )

    print(
        f"TOTAL: "
        f"{total_time:.2f}s"
    )

    return {
        "answer": answer,
        "sources": sources,
        "queries": queries,
        "metrics": {
            "query_generation": query_time,
            "retrieval": retrieval_time,
            "reranking": rerank_time,
            "compression": compression_time,
            "final_answer": answer_time,
            "total": total_time,
        },
    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    question = input(
        "Ask a question: "
    )

    asyncio.run(
        run_advanced_rag(question)
    )
    