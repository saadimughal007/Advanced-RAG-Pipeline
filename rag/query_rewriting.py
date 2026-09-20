import warnings

# Message match regex pattern
warnings.filterwarnings(
    action="ignore",
    message=r".*Direct use of automatic function calling.*",
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
    raise ValueError("HF_TOKEN is missing from the .env file.")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing from the .env file.")


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


# Retriever
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 5,
    }
)


# Gemini
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GOOGLE_API_KEY,
    thinking_level="low",
    max_output_tokens=500,
)


# Query rewriting prompt
rewrite_prompt = ChatPromptTemplate.from_template(
    """
    Rewrite the user's question into one clear search query
    optimized for retrieving relevant documents.

    Keep the original meaning.

    Remove unnecessary conversational words.

    Focus on important technical concepts and keywords.

    Return only the rewritten search query.

    User question:
    {question}
    """
)


# Convert model content to plain text
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


# Rewrite query
def rewrite_query(question: str):
    messages = rewrite_prompt.invoke({
        "question": question
    })

    response = model.invoke(messages)

    return extract_text(response.content).strip()


# Retrieve documents
def retrieve_documents(query: str):
    return retriever.invoke(query)


# Test
if __name__ == "__main__":

    question = input("Ask a question: ")

    rewritten_query = rewrite_query(question)

    print("\nOriginal Question:")
    print(question)

    print("\nRewritten Query:")
    print(rewritten_query)

    documents = retrieve_documents(rewritten_query)

    print("\nRetrieved Documents:")

    for index, document in enumerate(
        documents,
        start=1,
    ):
        print(f"\n--- Result {index} ---")
        print(document.page_content)

        print(
            "Source:",
            document.metadata.get("source")
        )