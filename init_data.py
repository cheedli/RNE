"""
Script to initialize the vector store and indices for the RNE chatbot.
This should be run once before starting the application.
Supports loading from multiple JSON files in a directory.
"""

import os
import sys
import traceback
import json
import glob

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_PATH, FAISS_INDEX_PATH, BM25_DATA_PATH
from preprocessing.data_loader import RNEDataLoader
from retrieval.faiss_retriever import FAISSRetriever
from retrieval.bm25_retriever import BM25Retriever

def verify_data_path(data_path):
    """Verify that the data path exists and contains valid JSON files."""
    if os.path.isfile(data_path):
        # Single file
        return verify_json_file(data_path)
    elif os.path.isdir(data_path):
        # Directory
        json_files = glob.glob(os.path.join(data_path, "*.json"))
        if not json_files:
            print(f"Error: No JSON files found in directory {data_path}")
            return False
        
        print(f"Found {len(json_files)} JSON files in {data_path}:")
        all_valid = True
        for json_file in sorted(json_files):
            print(f"  Checking {os.path.basename(json_file)}...")
            if not verify_json_file(json_file):
                all_valid = False
        
        return all_valid
    else:
        print(f"Error: Path not found: {data_path}")
        return False

def verify_json_file(file_path):
    """Verify that a JSON file exists and is valid."""
    if not os.path.exists(file_path):
        print(f"    Error: File not found at {file_path}")
        return False
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            print(f"    ✓ Valid JSON list with {len(data)} items")
            # Check the first item for structure
            if data:
                first_item = data[0]
                if isinstance(first_item, dict):
                    if "combined_content" in first_item:
                        print(f"    ✓ Detected combined_content format")
                    elif "code" in first_item:
                        print(f"    ✓ Detected RNE laws format")
                    else:
                        print(f"    ✓ Detected custom format")
                        
        elif isinstance(data, dict):
            print("    ✓ Valid JSON dictionary")
            if "combined_content" in data:
                print(f"    ✓ Detected combined_content format")
            elif "code" in data:
                print(f"    ✓ Detected RNE laws format")
            else:
                # Check if it contains lists that might be data
                has_lists = any(isinstance(v, list) for v in data.values())
                if has_lists:
                    print(f"    ✓ Detected structured format with lists")
                else:
                    print(f"    ✓ Detected custom format")
        else:
            print(f"    Warning: Unexpected JSON format. Expected list or dictionary.")
            
        return True
        
    except json.JSONDecodeError as e:
        print(f"    Error: Invalid JSON. Details: {str(e)}")
        return False
    except Exception as e:
        print(f"    Error: {str(e)}")
        return False

def initialize_indices():
    """Initialize FAISS and BM25 indices."""
    print("Initializing indices for the RNE chatbot...")
    print("=" * 50)
    
    # First verify the data path
    print(f"Verifying data at {DATA_PATH}...")
    if not verify_data_path(DATA_PATH):
        print("Cannot proceed with initialization due to data issues.")
        return False
    
    try:
        # Initialize data loader
        print(f"\nInitializing data loader...")
        data_loader = RNEDataLoader(DATA_PATH)
        
        # Load and process data
        print(f"\nLoading data...")
        data_loader.load_data()
        
        print(f"\nProcessing data...")
        processed_data = data_loader.process_data()
        
        # Show statistics
        print(f"\nData Statistics:")
        stats = data_loader.get_statistics()
        print(f"  Total documents: {stats['total_documents']}")
        print(f"  By language: {dict(stats['by_language'])}")
        print(f"  By source: {dict(stats['by_source'])}")
        if stats['by_type']:
            print(f"  By type: {dict(stats['by_type'])}")
        
        # Extract text for indexing
        print(f"\nExtracting text for indexing...")
        texts, docs = data_loader.extract_text_for_indexing()
        print(f"Extracted {len(texts)} text entries for indexing")
        
        if not texts:
            print("Error: No text extracted for indexing. Cannot proceed.")
            return False
        
        # Initialize retrievers
        print(f"\nInitializing retrievers...")
        faiss_retriever = FAISSRetriever(FAISS_INDEX_PATH)
        bm25_retriever = BM25Retriever(BM25_DATA_PATH)
        
        # Build indices
        print(f"\nBuilding FAISS index...")
        faiss_retriever.build_index(texts, docs)
        print(f"FAISS index saved to: {FAISS_INDEX_PATH}")
        
        print(f"\nBuilding BM25 index...")
        bm25_retriever.build_index(texts, docs)
        print(f"BM25 index saved to: {BM25_DATA_PATH}")
        
        print(f"\n✓ Indices built and saved successfully!")
        
        # Test retrieval with multiple queries
        test_queries = [
            "documents pour immatriculation sarl",
            "régime fiscal entreprise", 
            "création société anonyme",
            "obligations déclaration fiscale",
            "external_data business knowledge",
            "patente entreprise individuelle",
            "tva commerce détail"
        ]
        
        print(f"\nTesting retrieval system...")
        print("=" * 30)
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            
            try:
                # Test FAISS
                print("  FAISS results:")
                faiss_results = faiss_retriever.search(query, top_k=2)
                for i, result in enumerate(faiss_results):
                    doc_id = result['document'].get('id', 'unknown')
                    code = result['document'].get('code', 'unknown')
                    source = result['document'].get('source', 'unknown')
                    score = result['score']
                    print(f"    {i+1}. {doc_id} ({code}) from {source} - Score: {score:.4f}")
                    
                # Test BM25
                print("  BM25 results:")
                bm25_results = bm25_retriever.search(query, top_k=2)
                for i, result in enumerate(bm25_results):
                    doc_id = result['document'].get('id', 'unknown')
                    code = result['document'].get('code', 'unknown')
                    source = result['document'].get('source', 'unknown')
                    score = result['score']
                    print(f"    {i+1}. {doc_id} ({code}) from {source} - Score: {score:.4f}")
                    
            except Exception as e:
                print(f"  Error testing query '{query}': {str(e)}")
                
        print(f"\n" + "=" * 50)
        print("✓ Initialization complete!")
        print(f"✓ Total documents indexed: {len(texts)}")
        print(f"✓ FAISS index: {FAISS_INDEX_PATH}")
        print(f"✓ BM25 index: {BM25_DATA_PATH}")
        
        return True
        
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        return False

def ensure_directories_exist():
    """Ensure all necessary directories exist."""
    # Create data directory if it doesn't exist
    if os.path.isfile(DATA_PATH):
        data_dir = os.path.dirname(DATA_PATH)
    else:
        data_dir = DATA_PATH
        
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

def clean_indices():
    """Clean existing indices (useful for rebuilding)."""
    print("Cleaning existing indices...")
    
    # Remove FAISS index files
    faiss_files = [
        FAISS_INDEX_PATH + ".index",
        FAISS_INDEX_PATH + ".pkl"
    ]
    
    for file_path in faiss_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"  Removed: {file_path}")
            
    # Remove BM25 index file
    if os.path.exists(BM25_DATA_PATH):
        os.remove(BM25_DATA_PATH)
        print(f"  Removed: {BM25_DATA_PATH}")
        
    print("✓ Indices cleaned")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize RNE chatbot indices")
    parser.add_argument("--clean", action="store_true", help="Clean existing indices before rebuilding")
    parser.add_argument("--verify-only", action="store_true", help="Only verify data files without building indices")
    
    args = parser.parse_args()
    
    # Ensure directories exist
    ensure_directories_exist()
    
    if args.verify_only:
        print("Verification mode - checking data files only...")
        if verify_data_path(DATA_PATH):
            print("✓ All data files are valid")
        else:
            print("✗ Some data files have issues")
    else:
        if args.clean:
            clean_indices()
            
        # Initialize indices
        success = initialize_indices()
        
        if success:
            print("\n🎉 Ready to start the chatbot!")
        else:
            print("\n❌ Initialization failed. Please check the errors above.")
            sys.exit(1)