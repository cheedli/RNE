"""
Module for loading and preprocessing RNE laws data from JSON files.
Supports loading from multiple files in a directory.
"""

import json
import os
import glob
from typing import Dict, List, Any, Tuple

class RNEDataLoader:
    """Class for loading and preprocessing RNE laws data from multiple files."""
    
    def __init__(self, data_path: str):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to a JSON file OR directory containing JSON files.
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        
    def _get_json_files(self) -> List[str]:
        """
        Get list of JSON files to process.
        
        Returns:
            List of JSON file paths.
        """
        if os.path.isfile(self.data_path):
            # Single file
            return [self.data_path]
        elif os.path.isdir(self.data_path):
            # Directory - find all JSON files
            json_files = glob.glob(os.path.join(self.data_path, "*.json"))
            if not json_files:
                raise FileNotFoundError(f"No JSON files found in directory {self.data_path}")
            return sorted(json_files)  # Sort for consistent ordering
        else:
            raise FileNotFoundError(f"Path not found: {self.data_path}")
    
    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load RNE laws data from JSON file(s).
        
        Returns:
            List of dictionaries containing the RNE laws data from all files.
        """
        json_files = self._get_json_files()
        all_data = []
        
        print(f"Found {len(json_files)} JSON file(s) to process:")
        for file_path in json_files:
            print(f"  - {os.path.basename(file_path)}")
        
        for file_path in json_files:
            try:
                print(f"\nLoading data from {os.path.basename(file_path)}...")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                
                # Handle different data structures
                if isinstance(file_data, list):
                    # List of items
                    all_data.extend(file_data)
                    print(f"  Loaded {len(file_data)} items from list")
                elif isinstance(file_data, dict):
                    # Check if it's a single item or contains a list
                    if any(isinstance(v, list) for v in file_data.values()):
                        # Might be structured data with lists inside
                        # Try to extract items from lists
                        for key, value in file_data.items():
                            if isinstance(value, list):
                                all_data.extend(value)
                                print(f"  Loaded {len(value)} items from key '{key}'")
                            elif isinstance(value, dict):
                                all_data.append(value)
                                print(f"  Loaded 1 item from key '{key}'")
                    else:
                        # Single dictionary item
                        all_data.append(file_data)
                        print(f"  Loaded 1 item (single dict)")
                else:
                    print(f"  Warning: Unexpected data type in {file_path}: {type(file_data)}")
                    
            except json.JSONDecodeError as e:
                error_message = f"Error parsing JSON file {file_path}: {str(e)}"
                print(f"  Error: {error_message}")
                # Continue with other files instead of failing completely
                continue
            except Exception as e:
                error_message = f"Unexpected error loading data from {file_path}: {str(e)}"
                print(f"  Error: {error_message}")
                continue
        
        self.raw_data = all_data
        
        if not self.raw_data:
            print(f"Warning: No data loaded from any files in {self.data_path}")
        else:
            print(f"\nSuccessfully loaded {len(self.raw_data)} total items from all files")
            
        return self.raw_data
    
    def process_data(self) -> List[Dict[str, Any]]:
        """
        Process the raw RNE laws data into a format suitable for indexing.
        Handles multiple formats including external_data.json, combined_content, and legacy formats.
        
        Returns:
            List of processed documents for indexing.
        """
        if self.raw_data is None:
            self.load_data()
            
        processed_data = []
        
        for i, item in enumerate(self.raw_data):
            try:
                # Handle external_data.json format (business/fiscal knowledge)
                # Handle external_data.json format (business/fiscal knowledge)
                if "combined_content" in item or "combined_content_arabic" in item:
                    # Prefer French if available, else fallback to Arabic
                    content = item.get("combined_content") or item.get("combined_content_arabic")
                    
                    # Determine language
                    language = "fr" if "combined_content" in item else "ar"
                    
                    # Try to extract a meaningful ID from the content
                    content_preview = content[:100].lower()
                    doc_id = f"external_{i}"
                    doc_source = "external_data"
                    
                    # Try to categorize the content
                    if any(word in content_preview for word in ["impôt", "fiscal"]):
                        category = "fiscalité"
                    elif any(word in content_preview for word in ["entreprise", "société"]):
                        category = "création_entreprise"
                    elif any(word in content_preview for word in ["contrat", "juridique"]):
                        category = "juridique"
                    elif any(word in content_preview for word in ["employé", "travail"]):
                        category = "droit_travail"
                    else:
                        category = "général"
                    
                    doc = {
                        "id": doc_id,
                        "code": f"EXT_{i:03d}",
                        "language": language,
                        "type_entreprise": category,
                        "genre_entreprise": "guide_pratique",
                        "procedure": "information",
                        "redevance_demandee": "",
                        "delais": "",
                        "pdf_link": "",
                        "content": content,
                        "raw_content": item,
                        "source": doc_source,
                        "category": category
                    }
                    processed_data.append(doc)
                
                # Handle legacy structured format (RNE laws format)
                elif "code" in item:
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
                            "raw_content": item["french_content"],
                            "source": "rne_laws"
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
                            "raw_content": item["arabic_content"],
                            "source": "rne_laws"
                        }
                        processed_data.append(arabic_doc)
                        
                # Handle simple text content (fallback)
                else:
                    # Try to extract any text content from the item
                    text_content = self._extract_text_from_item(item)
                    if text_content:
                        doc = {
                            "id": f"item_{i}",
                            "code": f"ITEM_{i}",
                            "language": "fr",
                            "type_entreprise": "general",
                            "genre_entreprise": "guide",
                            "procedure": "informational",
                            "redevance_demandee": "",
                            "delais": "",
                            "pdf_link": "",
                            "content": text_content,
                            "raw_content": item,
                            "source": "misc"
                        }
                        processed_data.append(doc)
                        
            except Exception as e:
                print(f"Error processing item {i}: {str(e)}")
                continue
                
        self.processed_data = processed_data
        print(f"Processed {len(processed_data)} documents total")
        return processed_data
    
    def _extract_text_from_item(self, item: Any) -> str:
        """
        Extract text content from various item formats.
        
        Args:
            item: Item to extract text from.
            
        Returns:
            Extracted text content.
        """
        if isinstance(item, str):
            return item
        elif isinstance(item, dict):
            text_parts = []
            for key, value in item.items():
                if isinstance(value, str):
                    text_parts.append(f"{key}: {value}")
                elif isinstance(value, (list, dict)):
                    text_parts.append(f"{key}: {str(value)}")
            return "\n".join(text_parts)
        elif isinstance(item, list):
            return "\n".join([str(x) for x in item])
        else:
            return str(item)
    
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
    
    def get_documents_by_source(self, source: str) -> List[Dict[str, Any]]:
        """
        Retrieve all documents from a specific source.
        
        Args:
            source: Source identifier ('rne_laws', 'combined_knowledge', 'misc').
            
        Returns:
            List of documents from the specified source.
        """
        if self.processed_data is None:
            self.process_data()
            
        return [doc for doc in self.processed_data if doc.get("source") == source]
    
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
        else:
            print(f"Extracted {len(texts)} documents for indexing")
            
        return texts, docs
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded data.
        
        Returns:
            Dictionary containing statistics.
        """
        if self.processed_data is None:
            self.process_data()
            
        stats = {
            "total_documents": len(self.processed_data),
            "by_language": {},
            "by_source": {},
            "by_type": {}
        }
        
        for doc in self.processed_data:
            # Language stats
            lang = doc.get("language", "unknown")
            stats["by_language"][lang] = stats["by_language"].get(lang, 0) + 1
            
            # Source stats
            source = doc.get("source", "unknown")
            stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
            
            # Type stats
            doc_type = doc.get("type_entreprise", "unknown")
            stats["by_type"][doc_type] = stats["by_type"].get(doc_type, 0) + 1
            
        return stats