"""
Keyword-based retrieval system using BM25.
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi

import sys
sys.path.append('..')
from preprocessing.text_processor import TextProcessor

class BM25Retriever:
    """
    Keyword-based retrieval system using BM25 algorithm.
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the BM25 retriever.
        
        Args:
            data_path: Path to save/load the BM25 data.
        """
        self.data_path = data_path
        self.bm25_fr = None  # BM25 model for French documents
        self.bm25_ar = None  # BM25 model for Arabic documents
        self.tokenized_corpus_fr = []  # Tokenized French corpus
        self.tokenized_corpus_ar = []  # Tokenized Arabic corpus
        self.documents_fr = []  # French documents
        self.documents_ar = []  # Arabic documents
        self.text_processor = TextProcessor()
        
    def build_index(self, texts: List[str], documents: List[Dict[str, Any]]) -> None:
        """
        Build BM25 indices for the provided texts and documents.
        
        Args:
            texts: List of text strings to index.
            documents: List of document dictionaries corresponding to the texts.
        """
        # Separate documents by language
        texts_fr = []
        texts_ar = []
        self.documents_fr = []
        self.documents_ar = []
        
        for i, doc in enumerate(documents):
            if doc.get('language') == 'fr':
                texts_fr.append(texts[i])
                self.documents_fr.append(doc)
            elif doc.get('language') == 'ar':
                texts_ar.append(texts[i])
                self.documents_ar.append(doc)
        
        # Tokenize the corpus for each language
        print(f"Tokenizing {len(texts_fr)} French documents and {len(texts_ar)} Arabic documents...")
        self.tokenized_corpus_fr = [self.text_processor.preprocess(text, 'fr') for text in texts_fr]
        self.tokenized_corpus_ar = [self.text_processor.preprocess(text, 'ar') for text in texts_ar]
        
        # Create BM25 models
        self.bm25_fr = BM25Okapi(self.tokenized_corpus_fr) if self.tokenized_corpus_fr else None
        self.bm25_ar = BM25Okapi(self.tokenized_corpus_ar) if self.tokenized_corpus_ar else None
        
        print(f"Built BM25 indices for {len(self.tokenized_corpus_fr)} French and {len(self.tokenized_corpus_ar)} Arabic documents")
        
        # Save the data if a path is provided
        if self.data_path:
            self._save_data()
    
    def _save_data(self) -> None:
        """Save the BM25 data to disk."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        
        # Prepare data to save
        data = {
            'tokenized_corpus_fr': self.tokenized_corpus_fr,
            'tokenized_corpus_ar': self.tokenized_corpus_ar,
            'documents_fr': self.documents_fr,
            'documents_ar': self.documents_ar
        }
        
        # Save the data
        with open(self.data_path, 'wb') as f:
            pickle.dump(data, f)
            
        print(f"Saved BM25 data to {self.data_path}")
    
    def load_index(self) -> bool:
        """
        Load the BM25 data from disk and rebuild the models.
        
        Returns:
            True if successful, False otherwise.
        """
        if not self.data_path or not os.path.exists(self.data_path):
            print(f"BM25 data file not found at {self.data_path}")
            return False
            
        # Load the data
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
            
        # Extract the data
        self.tokenized_corpus_fr = data['tokenized_corpus_fr']
        self.tokenized_corpus_ar = data['tokenized_corpus_ar']
        self.documents_fr = data['documents_fr']
        self.documents_ar = data['documents_ar']
        
        # Rebuild the models
        self.bm25_fr = BM25Okapi(self.tokenized_corpus_fr) if self.tokenized_corpus_fr else None
        self.bm25_ar = BM25Okapi(self.tokenized_corpus_ar) if self.tokenized_corpus_ar else None
        
        print(f"Loaded BM25 indices with {len(self.tokenized_corpus_fr)} French and {len(self.tokenized_corpus_ar)} Arabic documents")
        return True
    
    def search(self, query: str, top_k: int = 3, language: str = None) -> List[Dict[str, Any]]:
        """
        Search the BM25 indices for documents relevant to the query.
        
        Args:
            query: Query string.
            top_k: Number of top results to return.
            language: Optional language filter ('fr' or 'ar'). If None, detect language from query.
            
        Returns:
            List of dictionaries with document info and scores.
        """
        # Detect language if not specified
        if not language:
            language = self.text_processor.detect_language(query)
            
        # Tokenize the query
        tokenized_query = self.text_processor.preprocess(query, language)
        
        results = []
        
        # Search in the appropriate language index
        if language == 'fr' and self.bm25_fr:
            scores = self.bm25_fr.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[-top_k:][::-1]  # Get top k indices in descending order
            
            for i, idx in enumerate(top_indices):
                if scores[idx] > 0:  # Only include results with positive scores
                    results.append({
                        'document': self.documents_fr[idx],
                        'score': float(scores[idx]),
                        'rank': i + 1
                    })
                    
        elif language == 'ar' and self.bm25_ar:
            scores = self.bm25_ar.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[-top_k:][::-1]
            
            for i, idx in enumerate(top_indices):
                if scores[idx] > 0:
                    results.append({
                        'document': self.documents_ar[idx],
                        'score': float(scores[idx]),
                        'rank': i + 1
                    })
        
        return results