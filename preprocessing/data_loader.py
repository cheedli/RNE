"""
Module for loading and preprocessing RNE laws data from JSON files.
"""

import json
import os
from typing import Dict, List, Any, Tuple

class RNEDataLoader:
    """Class for loading and preprocessing RNE laws data."""
    
    def __init__(self, data_path: str):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the JSON file containing RNE laws data.
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        
    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load RNE laws data from the JSON file.
        
        Returns:
            List of dictionaries containing the RNE laws data.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
                
            # If the data is a single dictionary, convert it to a list
            if isinstance(self.raw_data, dict):
                self.raw_data = [self.raw_data]
                
            # Validate the data structure
            if not self.raw_data:
                print(f"Warning: Empty data loaded from {self.data_path}")
            else:
                print(f"Successfully loaded {len(self.raw_data)} items from {self.data_path}")
                
            return self.raw_data
            
        except json.JSONDecodeError as e:
            error_message = f"Error parsing JSON file {self.data_path}: {str(e)}"
            print(error_message)
            raise ValueError(error_message)
        except Exception as e:
            error_message = f"Unexpected error loading data from {self.data_path}: {str(e)}"
            print(error_message)
            raise
    
    def process_data(self) -> List[Dict[str, Any]]:
        """
        Process the raw RNE laws data into a format suitable for indexing.
        
        Returns:
            List of processed documents for indexing.
        """
        if self.raw_data is None:
            self.load_data()
            
        processed_data = []
        
        for item in self.raw_data:
            # Process French content if available
            if "french_content" in item and item["french_content"]:
                french_doc = {
                    "id": f"{item['code']}_fr",
                    "code": item["code"],
                    "language": "fr",
                    "type_entreprise": item.get("type_entreprise", ""),
                    "genre_entreprise": item.get("genre_entreprise", ""),
                    "procedure": item.get("procedure", ""),
                    "redevance_demandee": item.get("redevance_demandee", ""),
                    "delais": item.get("delais", ""),
                    "pdf_link": item.get("pdf_french_link", ""),
                    "content": self._process_content(item["french_content"]),
                    "raw_content": item["french_content"]
                }
                processed_data.append(french_doc)
                
            # Process Arabic content if available
            if "arabic_content" in item and item["arabic_content"]:
                arabic_doc = {
                    "id": f"{item['code']}_ar",
                    "code": item["code"],
                    "language": "ar",
                    "type_entreprise": item.get("type_entreprise", ""),
                    "genre_entreprise": item.get("genre_entreprise", ""),
                    "procedure": item.get("procedure", ""),
                    "redevance_demandee": item.get("redevance_demandee", ""),
                    "delais": item.get("delais", ""),
                    "pdf_link": item.get("pdf_arabic_link", ""),
                    "content": self._process_content(item["arabic_content"]),
                    "raw_content": item["arabic_content"]
                }
                processed_data.append(arabic_doc)
                
        self.processed_data = processed_data
        return processed_data
    
    def _process_content(self, content: Dict[str, Any]) -> str:
        """
        Process content dictionary into a single string for indexing.
        
        Args:
            content: Dictionary containing content fields.
            
        Returns:
            Processed content as a single string.
        """
        # Handle case where content is None or not a dictionary
        if not content or not isinstance(content, dict):
            return ""
            
        processed_text = ""
        
        for key, value in content.items():
            if isinstance(value, list):
                processed_text += f"{key}: {' '.join(value)}\n"
            else:
                processed_text += f"{key}: {value}\n"
                
        return processed_text
    
    def get_document_by_id(self, doc_id: str) -> Dict[str, Any]:
        """
        Retrieve a document by its ID.
        
        Args:
            doc_id: Document ID to retrieve.
            
        Returns:
            Document dictionary or None if not found.
        """
        if self.processed_data is None:
            self.process_data()
            
        for doc in self.processed_data:
            if doc["id"] == doc_id:
                return doc
                
        return None
    
    def get_documents_by_code(self, code: str) -> List[Dict[str, Any]]:
        """
        Retrieve all documents matching a given RNE code.
        
        Args:
            code: RNE code to match.
            
        Returns:
            List of matching document dictionaries.
        """
        if self.processed_data is None:
            self.process_data()
            
        return [doc for doc in self.processed_data if doc["code"] == code]
    
    def get_documents_by_language(self, language: str) -> List[Dict[str, Any]]:
        """
        Retrieve all documents in a given language.
        
        Args:
            language: Language code ('fr' or 'ar').
            
        Returns:
            List of documents in the specified language.
        """
        if self.processed_data is None:
            self.process_data()
            
        return [doc for doc in self.processed_data if doc["language"] == language]
    
    def extract_text_for_indexing(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Extract text content for indexing and keep reference to original documents.
        
        Returns:
            Tuple containing (list of text contents, list of corresponding documents)
        """
        if self.processed_data is None:
            self.process_data()
            
        texts = []
        docs = []
        
        for doc in self.processed_data:
            try:
                # Combine all relevant fields for rich text representation
                text_parts = [
                    doc['code'],
                    doc.get('type_entreprise', ''),
                    doc.get('genre_entreprise', ''),
                    doc.get('procedure', ''),
                    doc.get('content', '')
                ]
                
                # Filter out empty parts
                text_parts = [part for part in text_parts if part]
                
                # Join all parts with spaces
                text = ' '.join(text_parts)
                
                # Only add documents with non-empty text
                if text.strip():
                    texts.append(text)
                    docs.append(doc)
                else:
                    print(f"Warning: Empty text for document {doc.get('id', 'unknown')}. Skipping.")
                    
            except Exception as e:
                print(f"Error processing document {doc.get('id', 'unknown')}: {str(e)}")
                # Continue with other documents
                continue
                
        if not texts:
            print("Warning: No text extracted for indexing. Check your data.")
            
        return texts, docs