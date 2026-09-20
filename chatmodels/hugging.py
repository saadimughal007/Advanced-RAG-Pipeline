import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

model = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4.1-Flash",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
)

chat_model = ChatHuggingFace(llm=model)

response = chat_model.invoke(
    "What is the future after gen AI, explain in  simple sentence."
)

print(response.content)


