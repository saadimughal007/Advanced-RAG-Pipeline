import warnings

warnings.filterwarnings(
    "ignore",
    message="Direct use of automatic function calling.*"
)

import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor



BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


HF_TOKEN = os.getenv("HF_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN is missing from the .env file."
    )

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is missing from the .env file."
    )


# --------------------------------------------------
# 1. Embedding Model
# --------------------------------------------------

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=HF_TOKEN,
)


# --------------------------------------------------
# 2. Load Chroma Vector Database
# --------------------------------------------------

vector_store = Chroma(
    collection_name="knowledge_base",
    embedding_function=embeddings,
    persist_directory=str(BASE_DIR / "chroma_db"),
)


# --------------------------------------------------
# 3. Create Base Retriever
# --------------------------------------------------

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 5,
    }
)


# --------------------------------------------------
# 4. Gemini Model
# --------------------------------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GOOGLE_API_KEY,
    thinking_level="low",
    max_output_tokens=500,
)


# --------------------------------------------------
# 5. Create Compressor
# --------------------------------------------------

compressor = LLMChainExtractor.from_llm(
    llm=model
)


# --------------------------------------------------
# 6. Create Compression Retriever
# --------------------------------------------------

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever,
)


# --------------------------------------------------
# 7. Run
# --------------------------------------------------

if __name__ == "__main__":

    question = input("Ask a question: ")

    print("\nOriginal Question:")
    print(question)

    # Normal retrieval
    retrieved_documents = retriever.invoke(
        question
    )

    print("\n==============================")
    print("BEFORE COMPRESSION")
    print("==============================")

    for index, document in enumerate(
        retrieved_documents,
        start=1,
    ):
        print(
            f"\n--- Document {index} ---"
        )

        print(document.page_content)

        print(
            "Source:",
            document.metadata.get("source")
        )


    # Contextual compression
    compressed_documents = (
        compression_retriever.invoke(
            question
        )
    )


    print("\n==============================")
    print("AFTER COMPRESSION")
    print("==============================")

    for index, document in enumerate(
        compressed_documents,
        start=1,
    ):
        print(
            f"\n--- Compressed Document {index} ---"
        )

        print(document.page_content)

        print(
            "Source:",
            document.metadata.get("source")
        )
        