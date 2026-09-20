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
from langchain_core.prompts import ChatPromptTemplate


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent


# Load environment variables
load_dotenv(BASE_DIR / ".env")


# API keys
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


# Embedding model
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=HF_TOKEN,
)


# Load Chroma
vector_store = Chroma(
    collection_name="knowledge_base",
    embedding_function=embeddings,
    persist_directory=str(BASE_DIR / "chroma_db"),
)


# MMR Retriever
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 2,
        "fetch_k": 10,
    },
)


# Gemini
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GOOGLE_API_KEY,
    thinking_level="low",
    max_output_tokens=500,
)


# Prompt for generating search queries
query_prompt = ChatPromptTemplate.from_template(
    """
    Generate 3 different search queries for the user's question.

    The queries should approach the question from different angles.

    Return only the 3 queries, one per line.

    User question:
    {question}
    """
)


# Convert Gemini content into plain text
def extract_text(content):

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


# Generate multiple search queries
def generate_queries(question: str):

    messages = query_prompt.invoke({
        "question": question
    })

    response = model.invoke(messages)

    content = extract_text(response.content)

    queries = [
        line.strip()
        for line in content.splitlines()
        if line.strip()
    ]

    return queries[:3]


# Remove exact duplicate documents
def remove_duplicates(documents):

    unique_documents = []
    seen = set()

    for document in documents:

        content = document.page_content

        if content not in seen:

            seen.add(content)
            unique_documents.append(document)

    return unique_documents


# Retrieve documents using multiple queries
def retrieve_documents(question: str):

    # Generate multiple search queries
    queries = generate_queries(question)

    # Store all retrieved documents
    all_documents = []

    # Search using every generated query
    for query in queries:

        documents = retriever.invoke(query)

        all_documents.extend(documents)

    # Remove duplicate documents
    unique_documents = remove_duplicates(
        all_documents
    )

    return queries, unique_documents


# Test
if __name__ == "__main__":

    question = input("Ask a question: ")

    queries, documents = retrieve_documents(question)

    print("\nGenerated Queries:")

    for index, query in enumerate(
        queries,
        start=1
    ):
        print(f"{index}. {query}")

    print("\nRetrieved Unique Documents:")

    for index, document in enumerate(
        documents,
        start=1
    ):

        print(f"\n--- Result {index} ---")

        print(document.page_content)

        print(
            "Source:",
            document.metadata.get("source")
        )
        