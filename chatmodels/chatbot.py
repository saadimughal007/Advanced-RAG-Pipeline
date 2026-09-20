import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# Environment Setup
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


# ============================================================
# Model
# ============================================================

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    thinking_level="low",
    max_output_tokens=500,
)


# ============================================================
# Prompt
# ============================================================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful technology-focused AI tutor.

        Follow these rules:

        1. Explain concepts in simple and easy-to-understand language.
        2. Keep answers concise.
        3. Focus mainly on AI, Python, automation, software development,
           LangChain, APIs, and related technology topics.
        4. If the user asks something unrelated to technology,
           politely explain that you are a technology-focused tutor.
        5. Do not make up information when you are unsure.
        """
    ),

    (
        "placeholder",
        "{history}"
    ),

    (
        "human",
        "{question}"
    ),
])


# ============================================================
# Chain
# ============================================================

chain = prompt | model
