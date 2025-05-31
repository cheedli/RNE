"""
Vector-based retrieval system using FAISS.
"""

import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

import sys
sys.path.append('..')
from config import EMBEDDING_MODEL, EMBEDDING_DIMENSION

class FAISSRetriever:
    """
    Vector-based retrieval system using FAISS for semantic search.
    """
    
    def __init__(self, index_path=None, embedding_model=EMBEDDING_MODEL):
        """
        Initialize the FAISS retriever.
        
        Args:
            index_path: Path to save/load the FAISS index.
            embedding_model: Name of the sentence transformer model to use.
        """
        self.index_path = index_path
        self.embedding_model = embedding_model
        self.index = None
        self.documents = []
        self.encoder = SentenceTransformer(embedding_model)
        
    def build_index(self, texts: List[str], documents: List[Dict[str, Any]]) -> None:
        """
        Build a FAISS index from the provided texts and documents.
        
        Args:
            texts: List of text strings to index.
            documents: List of document dictionaries corresponding to the texts.
        """
        # Store the documents for later retrieval
        self.documents = documents
        
        # Create embeddings for all texts
        print(f"Creating embeddings for {len(texts)} documents...")
        embeddings = self._create_embeddings(texts)
        
        # Create and configure the FAISS index
        dimension = EMBEDDING_DIMENSION
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity when normalized)
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add vectors to the index
        self.index.add(embeddings)
        print(f"Added {len(embeddings)} vectors to FAISS index")
        
        # Save the index if a path is provided
        if self.index_path:
            self._save_index()
    
    def _create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of text strings to encode.
            
        Returns:
            NumPy array of embeddings.
        """
        return self.encoder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    def _save_index(self) -> None:
        """Save the FAISS index and documents to disk."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Save the FAISS index
        faiss.write_index(self.index, self.index_path)
        
        # Save the documents in a separate file
        documents_path = self.index_path.replace('.bin', '_docs.pkl')
        with open(documents_path, 'wb') as f:
            pickle.dump(self.documents, f)
            
        print(f"Saved FAISS index to {self.index_path}")
    
    def load_index(self) -> bool:
        """
        Load the FAISS index and documents from disk.
        
        Returns:
            True if successful, False otherwise.
        """
        if not self.index_path or not os.path.exists(self.index_path):
            print(f"FAISS index file not found at {self.index_path}")
            return False
            
        # Load the FAISS index
        self.index = faiss.read_index(self.index_path)
        
        # Load the documents
        documents_path = self.index_path.replace('.bin', '_docs.pkl')
        if os.path.exists(documents_path):
            with open(documents_path, 'rb') as f:
                self.documents = pickle.load(f)
                
        print(f"Loaded FAISS index with {self.index.ntotal} vectors")
        return True
    
    def search(self, query: str, top_k: int = 3, language: str = None) -> List[Dict[str, Any]]:
        """
        Search the index for documents similar to the query.
        
        Args:
            query: Query string.
            top_k: Number of top results to return.
            language: Optional language filter ('fr' or 'ar').
            
        Returns:
            List of dictionaries with document info and scores.
        """
        if not self.index:
            print("FAISS index not built or loaded")
            return []
            
        # Create query embedding
        query_embedding = self._create_embeddings([query])
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search the index
        scores, indices = self.index.search(query_embedding, top_k * 2)  # Get more results for filtering
        
        # Extract results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self.documents):
                continue
                
            document = self.documents[idx]
            
            # Filter by language if specified
            if language and document.get('language') != language:
                continue
                
            results.append({
                'document': document,
                'score': float(score),  # Convert from numpy float to Python float
                'rank': i + 1
            })
            
            # Stop once we have enough results after filtering
            if len(results) >= top_k:
                break
                
        return results