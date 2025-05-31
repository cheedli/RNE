"""
Script to initialize the vector store and indices for the RNE chatbot.
This should be run once before starting the application.
"""

import os
import sys
import traceback
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_PATH, FAISS_INDEX_PATH, BM25_DATA_PATH
from preprocessing.data_loader import RNEDataLoader
from retrieval.faiss_retriever import FAISSRetriever
from retrieval.bm25_retriever import BM25Retriever

def verify_json_file(file_path):
    """Verify that the JSON file exists and is valid."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return False
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            print(f"JSON file is valid and contains {len(data)} items")
            # Check the first item for expected structure
            if data:
                first_item = data[0]
                expected_keys = ['code', 'type_entreprise', 'procedure']
                missing_keys = [key for key in expected_keys if key not in first_item]
                if missing_keys:
                    print(f"Warning: First item is missing expected keys: {missing_keys}")
                    
        elif isinstance(data, dict):
            print("JSON file is valid and contains a single item")
            expected_keys = ['code', 'type_entreprise', 'procedure']
            missing_keys = [key for key in expected_keys if key not in data]
            if missing_keys:
                print(f"Warning: Item is missing expected keys: {missing_keys}")
        else:
            print(f"Warning: JSON file has unexpected format. Expected a list or dictionary.")
            
        return True
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file at {file_path}. Details: {str(e)}")
        return False
    except Exception as e:
        print(f"Error checking JSON file: {str(e)}")
        return False

def initialize_indices():
    """Initialize FAISS and BM25 indices."""
    print("Initializing indices for the RNE chatbot...")
    
    # First verify the JSON file
    print(f"Verifying JSON file at {DATA_PATH}...")
    if not verify_json_file(DATA_PATH):
        print("Cannot proceed with initialization due to JSON file issues.")
        return
    
    try:
        # Initialize data loader
        data_loader = RNEDataLoader(DATA_PATH)
        
        # Load and process data
        print(f"Loading data from {DATA_PATH}...")
        data_loader.load_data()
        
        print("Processing data...")
        processed_data = data_loader.process_data()
        print(f"Processed {len(processed_data)} documents")
        
        # Extract text for indexing
        print("Extracting text for indexing...")
        texts, docs = data_loader.extract_text_for_indexing()
        print(f"Extracted {len(texts)} text entries for indexing")
        
        # Initialize retrievers
        faiss_retriever = FAISSRetriever(FAISS_INDEX_PATH)
        bm25_retriever = BM25Retriever(BM25_DATA_PATH)
        
        # Build indices
        print("Building FAISS index...")
        faiss_retriever.build_index(texts, docs)
        
        print("Building BM25 index...")
        bm25_retriever.build_index(texts, docs)
        
        print("Indices built and saved successfully!")
        
        # Test retrieval
        test_query = "documents pour immatriculation sarl"
        
        print("\nTesting retrieval with query:", test_query)
        
        # Test FAISS
        print("\nFAISS results:")
        faiss_results = faiss_retriever.search(test_query, top_k=2)
        for i, result in enumerate(faiss_results):
            print(f"Result {i+1}: {result['document']['code']} (Score: {result['score']:.4f})")
            
        # Test BM25
        print("\nBM25 results:")
        bm25_results = bm25_retriever.search(test_query, top_k=2)
        for i, result in enumerate(bm25_results):
            print(f"Result {i+1}: {result['document']['code']} (Score: {result['score']:.4f})")
            
        print("\nInitialization complete!")
        
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        print("Traceback:")
        traceback.print_exc()

def ensure_directories_exist():
    """Ensure all necessary directories exist."""
    # Create data directory if it doesn't exist
    data_dir = os.path.dirname(DATA_PATH)
    if not os.path.exists(data_dir):
        print(f"Creating data directory: {data_dir}")
        os.makedirs(data_dir)
        
    # Create directory for FAISS index if it doesn't exist
    faiss_dir = os.path.dirname(FAISS_INDEX_PATH)
    if not os.path.exists(faiss_dir):
        print(f"Creating FAISS index directory: {faiss_dir}")
        os.makedirs(faiss_dir)
        
    # Create directory for BM25 data if it doesn't exist
    bm25_dir = os.path.dirname(BM25_DATA_PATH)
    if not os.path.exists(bm25_dir):
        print(f"Creating BM25 data directory: {bm25_dir}")
        os.makedirs(bm25_dir)

if __name__ == "__main__":
    # Ensure directories exist
    ensure_directories_exist()
    
    # Initialize indices
    initialize_indices()