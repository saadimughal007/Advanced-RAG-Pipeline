import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv(BASE_DIR / ".env")


# API token
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing from the .env file.")


# Documents directory
DOCUMENTS_DIR = BASE_DIR / "documents"

if not DOCUMENTS_DIR.exists():
    raise FileNotFoundError(
        f"Documents directory not found: {DOCUMENTS_DIR}"
    )


# Load documents
loader = DirectoryLoader(
    str(DOCUMENTS_DIR),
    glob="**/*.txt",
    loader_cls=TextLoader,
)

documents = loader.load()

print(f"Loaded documents: {len(documents)}")


# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

chunks = splitter.split_documents(documents)

print(f"Created chunks: {len(chunks)}")


# Create embedding model
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=HF_TOKEN,
)


# Create Chroma vector store
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="knowledge_base",
    persist_directory=str(BASE_DIR / "chroma_db"),
)


print("Documents successfully stored in Chroma.")
print("Ingestion complete.")
