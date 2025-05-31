"""
Utilities for formatting chatbot responses.
"""

from typing import List, Dict, Any, Optional

class ResponseFormatter:
    """
    Class for formatting chatbot responses based on language and query type.
    """
    
    @staticmethod
    def format_response(
        response: str, 
        question: str, 
        documents: List[Dict[str, Any]], 
        language: str = 'fr'
    ) -> Dict[str, Any]:
        """
        Format the final response for the API output.
        
        Args:
            response: Generated response from the LLM.
            question: Original user question.
            documents: Retrieved documents used for context.
            language: Language code ('fr' or 'ar').
            
        Returns:
            Dictionary with formatted response data.
        """
        # Extract any referenced codes from the response
        referenced_codes = ResponseFormatter._extract_rne_codes(response)
        
        # Format document references
        references = []
        for doc in documents:
            document = doc['document']
            if document['code'] in referenced_codes:
                references.append({
                    'code': document['code'],
                    'procedure': document.get('procedure', ''),
                    'score': doc['score'],
                    'pdf_link': document.get('pdf_link', '')
                })
        
        # Determine text direction based on language
        text_direction = 'rtl' if language == 'ar' else 'ltr'
        
        return {
            'response': response,
            'language': language,
            'text_direction': text_direction,
            'references': references,
            'query': question,
            'document_count': len(documents)
        }
    
    @staticmethod
    def format_multi_response(
        responses: List[Dict[str, Any]],
        original_query: str,
        language: str = 'fr'
    ) -> Dict[str, Any]:
        """
        Format responses for multiple questions in a single query.
        
        Args:
            responses: List of response dictionaries for each question.
            original_query: Original user query containing multiple questions.
            language: Language code ('fr' or 'ar').
            
        Returns:
            Dictionary with formatted combined response data.
        """
        # Combine all responses
        combined_response = ""
        all_references = []
        total_docs = 0
        
        for i, resp in enumerate(responses):
            question = resp.get('query', '')
            answer = resp.get('response', '')
            
            # Add question and answer
            if language == 'fr':
                combined_response += f"**Question {i+1}:** {question}\n\n"
                combined_response += f"**Réponse {i+1}:** {answer}\n\n"
                combined_response += "---\n\n"
            else:
                combined_response += f"**السؤال {i+1}:** {question}\n\n"
                combined_response += f"**الإجابة {i+1}:** {answer}\n\n"
                combined_response += "---\n\n"
                
            # Collect references
            all_references.extend(resp.get('references', []))
            total_docs += resp.get('document_count', 0)
            
        # Remove duplicates from references by code
        unique_refs = {}
        for ref in all_references:
            code = ref['code']
            if code not in unique_refs or ref['score'] > unique_refs[code]['score']:
                unique_refs[code] = ref
                
        unique_references = list(unique_refs.values())
        
        # Determine text direction based on language
        text_direction = 'rtl' if language == 'ar' else 'ltr'
        
        return {
            'response': combined_response.strip(),
            'language': language,
            'text_direction': text_direction,
            'references': unique_references,
            'query': original_query,
            'question_count': len(responses),
            'document_count': total_docs
        }
    
    @staticmethod
    def _extract_rne_codes(text: str) -> List[str]:
        """
        Extract RNE codes from a text.
        
        Args:
            text: Text to extract codes from.
            
        Returns:
            List of extracted RNE codes.
        """
        import re
        
        # Pattern to match RNE codes (e.g., RNE M 004.37)
        pattern = r'RNE\s+[A-Z]\s+\d+\.\d+'
        
        # Find all matches
        matches = re.findall(pattern, text)
        
        # Remove duplicates
        unique_codes = list(set(matches))
        
        return unique_codes