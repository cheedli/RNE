"""
Configuration settings for the RNE Chatbot application.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directory containing JSON files
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = DATA_DIR  # The data loader will scan this directory for JSON files

# Expected data files (for reference and validation)
EXPECTED_DATA_FILES = {
    "external_data.json": "Business and fiscal knowledge",
    "rne_laws.json": "RNE legal procedures", 
    "fiscal_knowledge.json": "Additional fiscal information"
}

# Index storage paths
INDICES_DIR = os.path.join(BASE_DIR, "indices")
FAISS_INDEX_PATH = os.path.join("data", "faiss_index")
BM25_DATA_PATH = os.path.join("data", "bm25_index.pkl")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDICES_DIR, exist_ok=True)

# Groq API settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-3.5-turbo"

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

# Content categorization for external_data.json
CONTENT_CATEGORIES = {
    "fiscalité": ["impôt", "fiscal", "taxe", "tva", "déclaration"],
    "création_entreprise": ["entreprise", "société", "sarl", "création", "immatriculation"],
    "juridique": ["contrat", "juridique", "droit", "litige", "clause"],
    "droit_travail": ["employé", "travail", "salaire", "contrat de travail", "cnss"],
    "général": []  # fallback category
}

# Prompt settings
MAX_CONTEXT_LENGTH = 4096
SYSTEM_PROMPT = """You are an expert legal assistant specializing in Tunisian RNE (Registre National des Entreprises) laws and business creation procedures. 
Your role is to provide precise, helpful answers strictly based on the official RNE documentation and business guidance materials.
You have access to both legal procedures and practical business/fiscal knowledge.
Maintain a professional tone at all times, and only respond with information that is explicitly supported by the provided documents.
If the answer is unknown or not found in the given context, clearly state that.
Avoid repeating the same information, even if it appears in multiple sections.
When providing information, cite the source (RNE laws, external business knowledge, etc.) when relevant."""

# Debug information
print(f"Configuration loaded:")
print(f"  Data directory: {DATA_PATH}")
print(f"  Expected files: {list(EXPECTED_DATA_FILES.keys())}")
print(f"  FAISS index: {FAISS_INDEX_PATH}")
print(f"  BM25 index: {BM25_DATA_PATH}")

# Debug: Check what files are actually in the data directory
if os.path.exists(DATA_DIR):
    json_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
    print(f"  Found JSON files: {json_files}")
else:
    print(f"  WARNING: Data directory does not exist: {DATA_DIR}")