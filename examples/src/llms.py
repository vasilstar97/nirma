import os
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv('CHAT_MODEL'),
    base_url=os.getenv('BASE_URL') + '/v1',
    temperature=os.getenv('TEMPERATURE'),
    api_key=os.getenv('BASE_URL')
)

embedding = OllamaEmbeddings(
    model=os.getenv('EMBEDDING_MODEL'), 
    base_url=os.getenv('BASE_URL')
)

__all__ = [
    'llm',
    'embedding',
]