"""
Integration with OpenAI API for LLM capabilities.
"""

import openai
import json
from typing import Dict, List, Any, Optional

import sys
sys.path.append('..')
from config import OPENAI_API_KEY, LLM_MODEL, SYSTEM_PROMPT, MAX_CONTEXT_LENGTH

class OpenAIClient:
    """
    Client for interacting with OpenAI's LLM API.
    """
    
    def __init__(self, api_key: str = OPENAI_API_KEY, model: str = LLM_MODEL):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key.
            model: Model identifier to use (e.g., 'gpt-4', 'gpt-3.5-turbo').
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.system_prompt = SYSTEM_PROMPT
        
    def generate_response(
        self, 
        query: str, 
        context: List[Dict[str, Any]], 
        language: str = 'fr',
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a response using the OpenAI LLM.
        
        Args:
            query: User query.
            context: List of retrieved documents for context.
            language: Language to respond in ('fr' or 'ar').
            system_prompt: Optional custom system prompt.
            
        Returns:
            Generated response from the LLM.
        """
        # Use custom system prompt if provided, otherwise use default
        if system_prompt:
            prompt = system_prompt
        else:
            prompt = self._get_system_prompt(language)
            
        # Format context for the prompt
        formatted_context = self._format_context(context)
        
        # Prepare messages for the chat completion
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Contexte:\n{formatted_context}\n\nQuestion: {query}"}
        ]
        
        try:
            # Call the OpenAI API
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,  # Low temperature for more deterministic responses
                max_tokens=1024,
                top_p=0.9,
                stream=False
            )
            
            # Extract and return the response
            response = completion.choices[0].message.content
            return response
            
        except Exception as e:
            print(f"Error calling OpenAI API: {str(e)}")
            return f"Désolé, je n'ai pas pu générer une réponse. Erreur: {str(e)}"
    
    def segment_questions(self, query: str) -> List[str]:
        """
        Use the LLM to segment a query into multiple questions if needed.
        
        Args:
            query: User query that may contain multiple questions.
            
        Returns:
            List of individual questions.
        """
        prompt = """
Divise uniquement les phrases contenant plusieurs questions en questions distinctes.
Ne transforme pas des sujets ou titres en questions. Ne fais pas de brainstorming.
Retourne une liste de questions, une par ligne. Si le texte contient une seule question, retourne-la telle quelle.

Exemples :

Texte : "Quel est le délai de création d'une SARL et quelles sont les pièces à fournir ?"
Sortie :
Quel est le délai de création d'une SARL ?
Quelles sont les pièces à fournir ?

Texte : "Quels sont les frais pour créer une entreprise individuelle ?"
Sortie :
Quels sont les frais pour créer une entreprise individuelle ?

Texte : "Création du SARL et checklist"
Sortie :
Création du SARL et checklist

Texte : "Quel est le Délais de la Création Personne physique commerçant et quel sont les Redevances à acquitter pour tout type d'entreprise ?"
Sortie :
Quel est le Délais de la Création Personne physique commerçant ?
Quel sont les Redevances à acquitter pour tout type d'entreprise ?

Texte : """


        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=512,
                stream=False
            )
            
            response = completion.choices[0].message.content
            
            # Split the response into lines and filter out empty lines
            questions = [q.strip() for q in response.split('\n') if q.strip()]
            
            # If no questions were detected, return the original query
            if not questions:
                return [query]
                
            return questions
            
        except Exception as e:
            print(f"Error in question segmentation: {str(e)}")
            # On error, just return the original query
            return [query]
    
    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into a string for the LLM context.
        
        Args:
            context: List of retrieved documents.
            
        Returns:
            Formatted context string.
        """
        if not context:
            return "Aucun contexte pertinent trouvé."
            
        formatted_text = ""
        
        for i, item in enumerate(context):
            doc = item['document']
            score = item['score']
            
            formatted_text += f"--- Document {i+1} (Pertinence: {score:.2f}) ---\n"
            formatted_text += f"Code: {doc['code']}\n"
            formatted_text += f"Type d'entreprise: {doc['type_entreprise']}\n"
            formatted_text += f"Genre d'entreprise: {doc['genre_entreprise']}\n"
            formatted_text += f"Procédure: {doc['procedure']}\n"
            formatted_text += f"Redevance demandée: {doc['redevance_demandee']}\n"
            formatted_text += f"Délais: {doc['delais']}\n"
            
            # Add the raw content
            raw_content = doc.get('raw_content', {})
            if raw_content:
                formatted_text += "Contenu détaillé:\n"
                
                for key, value in raw_content.items():
                    if isinstance(value, list):
                        formatted_text += f"{key}: {', '.join(value)}\n"
                    else:
                        formatted_text += f"{key}: {value}\n"
                        
            formatted_text += f"Lien PDF: {doc.get('pdf_link', 'Non disponible')}\n\n"
            
        # Ensure we don't exceed the context limit
        if len(formatted_text) > MAX_CONTEXT_LENGTH:
            formatted_text = formatted_text[:MAX_CONTEXT_LENGTH] + "...[Contexte tronqué]"
            
        return formatted_text
    
    def _get_system_prompt(self, language: str) -> str:
        """
        Get the appropriate system prompt based on the language.
        
        Args:
            language: Language code ('fr' or 'ar').
            
        Returns:
            System prompt in the specified language.
        """
        if language == 'ar':
            return """إنت معاون قانوني مختص في قوانين السجل الوطني للمؤسسات (RNE) في تونس.
    خدمتك إنك تجاوب على الأسئلة وتعطي معلومات صحيحة، مبنية على الوثائق الرسمية متاع السجل.
    لازمك ديما تحكي مع الناس باللهجة التونسية، بالدارجة، باش تكون أقرب ليهم وأسهل للفهم.
    حافظ على نبرة محترفة، وما تعطي كان المعلومة اللي مأكدة وموثقة.
    وإذا المعلومة ما هيش موجودة، ولا ماكش متأكد، قول هذا بكل وضوح."""

        else:
            return self.system_prompt