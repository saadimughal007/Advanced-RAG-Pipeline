import warnings
warnings.filterwarnings("ignore", message="Direct use of automatic function calling.*")

import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

# 1. SETUP
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not HF_TOKEN or not GOOGLE_API_KEY:
    raise ValueError("Missing API Keys in environment variables.")

# Components Initialization
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=HF_TOKEN,
)

vector_store = Chroma(
    collection_name="knowledge_base",
    embedding_function=embeddings,
    persist_directory=str(BASE_DIR / "chroma_db"),
)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 10},
)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.0,
)

reranker = CrossEncoder("BAAI/bge-reranker-base")

# 2. PROMPTS & UTILS
rewrite_prompt = ChatPromptTemplate.from_template("""
Rewrite the user's question into ONE clear search query optimized for retrieving relevant documents.
Keep the original meaning, remove conversational words, and focus on technical concepts.
Return only the rewritten search query. No extra text.

User question: {question}
""")

multi_query_prompt = ChatPromptTemplate.from_template("""
Generate 3 different search queries for the user's question.
Return only the 3 queries, one per line. No numbering or extra text.

User question: {question}
""")

compress_prompt = ChatPromptTemplate.from_template("""
Extract only the relevant information from the given text that answers the user's question.
Question: {question}
Text: {text}
Compressed text:
""")

final_prompt = ChatPromptTemplate.from_template("""
Answer the user's question using ONLY the provided context.
Context:
{context}

Question:
{question}

Answer:
""")

def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join([p if isinstance(p, str) else p.get("text", "") for p in content if isinstance(p, (str, dict))])
    return str(content)

# 3. FASTAPI APP & SCHEMAS
app = FastAPI(
    title="Production 5-Stage Advanced RAG API",
    version="1.0.0",
    description="Multi-query, CrossEncoder Reranked, Compressed RAG Pipeline",
)

class QueryRequest(BaseModel):
    question: str = Field(..., example="How do vector databases handle similarity search?")

class QueryResponse(BaseModel):
    answer: str
    sources: list
    pipeline_info: dict

# 4. PIPELINE LOGIC (ASYNC)
async def run_advanced_rag_pipeline(question: str) -> dict:
    # Stage 1: Query Rewriting
    messages = rewrite_prompt.invoke({"question": question})
    response = await model.ainvoke(messages)
    rewritten = extract_text(response.content).strip()

    # Stage 2: Multi-Query Retrieval
    mq_messages = multi_query_prompt.invoke({"question": question})
    mq_response = await model.ainvoke(mq_messages)
    queries = [line.strip() for line in extract_text(mq_response.content).splitlines() if line.strip()][:3]

    all_docs = []
    for q in queries:
        docs = retriever.invoke(q)
        all_docs.extend(docs)

    # Deduplication
    unique_docs = []
    seen = set()
    for doc in all_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    # Stage 3: Reranking
    pairs = [(question, doc.page_content) for doc in unique_docs]
    scores = reranker.predict(pairs) if pairs else []
    ranked_docs = sorted(zip(scores, unique_docs), key=lambda x: x[0], reverse=True)[:3]

    # Stage 4: Context Compression
    compressed_docs = []
    for score, doc in ranked_docs:
        c_msg = compress_prompt.invoke({"question": question, "text": doc.page_content})
        c_res = await model.ainvoke(c_msg)
        compressed_text = extract_text(c_res.content).strip()
        compressed_docs.append(Document(page_content=compressed_text, metadata=doc.metadata))

    # Stage 5: Final Answer Generation
    context_parts = [f"[Source {i+1}: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}" for i, doc in enumerate(compressed_docs)]
    context = "\n\n".join(context_parts)

    f_msg = final_prompt.invoke({"context": context, "question": question})
    f_res = await model.ainvoke(f_msg)
    answer = extract_text(f_res.content).strip()

    sources = [doc.metadata.get("source", "Unknown") for doc in compressed_docs]

    return {
        "answer": answer,
        "sources": sources,
        "pipeline_info": {
            "original_question": question,
            "rewritten_query": rewritten,
            "generated_queries": queries,
            "total_retrieved": len(unique_docs),
            "reranked_top_k": len(ranked_docs),
            "compressed_documents": len(compressed_docs),
        }
    }

@app.post("/api/v1/rag/query", response_model=QueryResponse)
async def rag_query_endpoint(payload: QueryRequest):
    try:
        result = await run_advanced_rag_pipeline(payload.question)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG Pipeline Error: {str(e)}")

@app.get("/")
async def root():
    return {"status": "online", "system": "5-Stage Advanced RAG Microservice"}
