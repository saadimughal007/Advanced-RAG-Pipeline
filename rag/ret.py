import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv(BASE_DIR / ".env")


# Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing from the .env file.")


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


# Create MMR retriever
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10,
    },
)


# Ask a question
question = input("Ask a question: ")


# Retrieve relevant and diverse documents
results = retriever.invoke(question)


# Display results
print("\nRetrieved Documents:")

for index, document in enumerate(results, start=1):
    print(f"\n--- Result {index} ---")
    print(document.page_content)
    print("Source:", document.metadata.get("source"))