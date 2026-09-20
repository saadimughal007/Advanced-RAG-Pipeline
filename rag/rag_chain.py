import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv(BASE_DIR / ".env")


# API keys
HF_TOKEN = os.getenv("HF_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing from the .env file.")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing from the .env file.")


# Embedding model
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=HF_TOKEN,
)


# Load existing Chroma database
vector_store = Chroma(
    collection_name="knowledge_base",
    embedding_function=embeddings,
    persist_directory=str(BASE_DIR / "chroma_db"),
)


# Create retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# LLM
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GOOGLE_API_KEY,
    thinking_level="low",
    max_output_tokens=500,
)


# Prompt
prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful AI assistant.

    Answer the user's question using only the provided context.

    If the answer is not present in the context,
    say that you do not have enough information.

    Context:
    {context}

    Question:
    {question}
    """
)


# Convert retrieved documents into plain text
def format_documents(documents):
    return "\n\n".join(
        document.page_content
        for document in documents
    )


# Complete RAG function
def ask_rag(question: str):
    documents = retriever.invoke(question)

    context = format_documents(documents)

    messages = prompt.invoke({
        "context": context,
        "question": question,
    })

    response = model.invoke(messages)

    return {
        "answer": response.content,
        "sources": [
            document.metadata.get("source")
            for document in documents
        ],
    }


# Test
if __name__ == "__main__":
    question = input("Ask a question: ")

    result = ask_rag(question)

    print("\nAI:")
    print(result["answer"])

    print("\nSources:")

    for source in result["sources"]:
        print(source)
        