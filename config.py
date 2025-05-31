"""
Configuration settings for the RNE Chatbot application.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Groq API settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"

# Retrieval settings
FAISS_WEIGHT = 0.5  # Weight for FAISS retrieval (semantic search)
BM25_WEIGHT = 0.5   # Weight for BM25 retrieval (keyword search)
TOP_K_RETRIEVAL = 3  # Number of documents to retrieve

# Vector embedding settings
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"  # Supports both French and Arabic
EMBEDDING_DIMENSION = 768

# Language settings
SUPPORTED_LANGUAGES = ["fr", "ar"]
DEFAULT_LANGUAGE = "fr"

# Flask settings
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 5000))

# Data paths
DATA_PATH = "data/rne_laws.json"
FAISS_INDEX_PATH = "data/faiss_index.bin"
BM25_DATA_PATH = "data/bm25_data.pkl"

# Prompt settings
MAX_CONTEXT_LENGTH = 4096
SYSTEM_PROMPT = """You are an expert legal assistant specializing in Tunisian RNE (Registre National des Entreprises) laws. 
Your role is to provide precise, helpful answers strictly based on the official RNE documentation.
Maintain a professional tone at all times, and only respond with information that is explicitly supported by the provided documents.
If the answer is unknown or not found in the given context, clearly state that.
Avoid repeating the same information, even if it appears in multiple sections."""
