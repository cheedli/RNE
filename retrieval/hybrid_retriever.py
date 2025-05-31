"""
Hybrid retrieval system combining FAISS and BM25.
"""

from typing import List, Dict, Any
import numpy as np

import sys
sys.path.append('..')
from config import FAISS_WEIGHT, BM25_WEIGHT, TOP_K_RETRIEVAL
from retrieval.faiss_retriever import FAISSRetriever
from retrieval.bm25_retriever import BM25Retriever
from preprocessing.text_processor import TextProcessor

class HybridRetriever:
    """
    Hybrid retrieval system combining semantic search (FAISS) and keyword search (BM25).
    """
    
    def __init__(
        self, 
        faiss_retriever: FAISSRetriever, 
        bm25_retriever: BM25Retriever,
        faiss_weight: float = FAISS_WEIGHT,
        bm25_weight: float = BM25_WEIGHT
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            faiss_retriever: Initialized FAISS retriever.
            bm25_retriever: Initialized BM25 retriever.
            faiss_weight: Weight for FAISS results in hybrid scoring.
            bm25_weight: Weight for BM25 results in hybrid scoring.
        """
        self.faiss_retriever = faiss_retriever
        self.bm25_retriever = bm25_retriever
        self.faiss_weight = faiss_weight
        self.bm25_weight = bm25_weight
        self.text_processor = TextProcessor()
        
    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL, language: str = None) -> List[Dict[str, Any]]:
        """
        Perform a hybrid search using both FAISS and BM25.
        
        Args:
            query: Query string.
            top_k: Number of top results to return.
            language: Optional language filter ('fr' or 'ar'). If None, detect language from query.
            
        Returns:
            List of dictionaries with document info and combined scores.
        """
        # Detect language if not specified
        if not language:
            language = self.text_processor.detect_language(query)
            
        # Increase the number of results to retrieve from each system to ensure we have enough for reranking
        internal_top_k = max(top_k * 2, 10)
        
        # Get results from both retrieval systems
        faiss_results = self.faiss_retriever.search(query, internal_top_k, language)
        bm25_results = self.bm25_retriever.search(query, internal_top_k, language)
        
        # Combine and rerank results
        combined_results = self._combine_results(faiss_results, bm25_results, top_k)
        
        return combined_results
    
    def _normalize_scores(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize scores to be between 0 and 1.
        
        Args:
            results: List of result dictionaries with scores.
            
        Returns:
            List of result dictionaries with normalized scores.
        """
        if not results:
            return []
            
        # Extract scores
        scores = [result['score'] for result in results]
        
        # Find min and max scores
        min_score = min(scores)
        max_score = max(scores)
        
        # Avoid division by zero
        if max_score == min_score:
            return results
            
        # Normalize scores
        for result in results:
            result['score'] = (result['score'] - min_score) / (max_score - min_score)
            
        return results
    
    def _combine_results(
        self, 
        faiss_results: List[Dict[str, Any]], 
        bm25_results: List[Dict[str, Any]], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Combine and rerank results from both retrieval systems.
        
        Args:
            faiss_results: Results from FAISS retriever.
            bm25_results: Results from BM25 retriever.
            top_k: Number of top results to return.
            
        Returns:
            List of combined and reranked results.
        """
        # Normalize scores within each system
        faiss_results = self._normalize_scores(faiss_results)
        bm25_results = self._normalize_scores(bm25_results)
        
        # Create a dictionary to store combined scores
        doc_scores = {}
        
        # Add FAISS scores
        for result in faiss_results:
            doc_id = result['document']['id']
            doc_scores[doc_id] = {
                'document': result['document'],
                'faiss_score': result['score'] * self.faiss_weight,
                'bm25_score': 0,
                'combined_score': 0
            }
            
        # Add BM25 scores
        for result in bm25_results:
            doc_id = result['document']['id']
            if doc_id in doc_scores:
                doc_scores[doc_id]['bm25_score'] = result['score'] * self.bm25_weight
            else:
                doc_scores[doc_id] = {
                    'document': result['document'],
                    'faiss_score': 0,
                    'bm25_score': result['score'] * self.bm25_weight,
                    'combined_score': 0
                }
                
        # Calculate combined scores
        for doc_id, data in doc_scores.items():
            data['combined_score'] = data['faiss_score'] + data['bm25_score']
            
        # Sort by combined score
        sorted_results = sorted(
            doc_scores.values(), 
            key=lambda x: x['combined_score'], 
            reverse=True
        )
        
        # Format the final results
        final_results = []
        for i, result in enumerate(sorted_results[:top_k]):
            final_results.append({
                'document': result['document'],
                'score': result['combined_score'],
                'faiss_score': result['faiss_score'],
                'bm25_score': result['bm25_score'],
                'rank': i + 1
            })
            
        return final_results