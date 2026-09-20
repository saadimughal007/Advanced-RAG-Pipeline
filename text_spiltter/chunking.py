from langchain_text_splitters import RecursiveCharacterTextSplitter


text = """
Python is a high-level programming language.

FastAPI is a modern Python framework for building APIs.

LangChain is a framework for developing applications powered by language models.

RAG stands for Retrieval-Augmented Generation. It allows an AI application to retrieve relevant information before generating an answer.

Vector databases store vector representations of data and allow similarity search.
"""


splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
)

chunks = splitter.split_text(text)


print(f"Total chunks: {len(chunks)}")

for index, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {index} ---")
    print(chunk)
    