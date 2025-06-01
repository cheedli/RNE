"""
Main Flask application for RNE Chatbot with enhanced vague question detection.
"""

import os
import json
import traceback
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Import configuration
from config import (
    FAISS_INDEX_PATH, 
    BM25_DATA_PATH,
    DATA_PATH, 
    TOP_K_RETRIEVAL,
    DEBUG,
    HOST,
    PORT
)

# Import components
from preprocessing.data_loader import RNEDataLoader
from preprocessing.text_processor import TextProcessor
from retrieval.faiss_retriever import FAISSRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from utils.language_detector import LanguageDetector
from utils.response_formatter import ResponseFormatter

# Enhanced OpenAI client with vague question detection
from llm.openai_client import OpenAIClient, ResponseType, FollowUpResponse, DirectResponse

# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
if DEBUG:
    logging.basicConfig(level=logging.DEBUG)

# Global variables for components
data_loader = None
faiss_retriever = None
bm25_retriever = None
hybrid_retriever = None
openai_client = None
language_detector = None
text_processor = None

def initialize_components():
    """Initialize all components of the RNE chatbot."""
    global data_loader, faiss_retriever, bm25_retriever, hybrid_retriever, openai_client, language_detector, text_processor
    
    try:
        # Initialize text processor and language detector
        text_processor = TextProcessor()
        language_detector = LanguageDetector()
        
        # Initialize data loader
        data_loader = RNEDataLoader(DATA_PATH)
        
        # Initialize retrievers
        faiss_retriever = FAISSRetriever(FAISS_INDEX_PATH)
        bm25_retriever = BM25Retriever(BM25_DATA_PATH)
        
        # Check if indices exist, otherwise build them
        indices_exist = faiss_retriever.load_index() and bm25_retriever.load_index()
        
        if not indices_exist:
            print("Building indices for retrieval systems...")
            
            # Load and process data
            data_loader.load_data()
            processed_data = data_loader.process_data()
            
            # Extract text for indexing
            texts, docs = data_loader.extract_text_for_indexing()
            
            # Build indices
            faiss_retriever.build_index(texts, docs)
            bm25_retriever.build_index(texts, docs)
        
        # Initialize hybrid retriever
        hybrid_retriever = HybridRetriever(faiss_retriever, bm25_retriever)
        
        # Initialize enhanced OpenAI client
        openai_client = OpenAIClient()
        
        print("All components initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing components: {str(e)}")
        print(traceback.format_exc())
        raise

@app.route('/')
def index():
    """Render the main page of the application."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Enhanced API endpoint for chatting with the RNE chatbot.
    
    Expects JSON with:
    - query OR message: User query text (supports both field names)
    - language: (Optional) Preferred language ('fr' or 'ar')
    - context: (Optional) Conversation context for follow-ups
    
    Returns:
    - JSON response with generated answer and metadata
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        # Support both 'query' and 'message' field names for compatibility
        user_query = data.get('query') or data.get('message', '')
        preferred_language = data.get('language', 'fr')
        conversation_context = data.get('context', {})
        
        # Validate input
        if not user_query:
            return jsonify({
                'success': False,
                'error': 'Missing query/message parameter'
            }), 400
        
        # Log the request for debugging
        if DEBUG:
            logging.info(f"Received chat request: query='{user_query[:100]}...', language={preferred_language}")
            
        # Detect language if not specified or if detection is needed
        if not preferred_language or preferred_language not in ['fr', 'ar']:
            preferred_language = language_detector.detect_language(user_query)
            
        # Process query with enhanced system
        response = process_enhanced_query(user_query, preferred_language, conversation_context)
        
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'response': {
                'type': 'error',
                'response': f'Erreur du serveur: {str(e)}'
            }
        }), 500

def process_enhanced_query(query, language, conversation_context):
    """
    Process a user query through the enhanced chatbot pipeline.
    
    Args:
        query: User query text.
        language: Language code ('fr' or 'ar').
        conversation_context: Context from previous conversation.
        
    Returns:
        Dict with response data.
    """
    try:
        # Check if this is a follow-up response (user selected an option)
        if conversation_context.get('awaiting_clarification'):
            return handle_follow_up_response(query, conversation_context, language)
        
        # Check if query contains multiple questions
        questions = openai_client.segment_questions(query)
        
        if len(questions) == 1:
            # Single question - use enhanced processing
            return process_single_question_enhanced(query, language)
        else:
            # Multiple questions - process each separately
            return process_multiple_questions(questions, language)
            
    except Exception as e:
        logging.error(f"Error in process_enhanced_query: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'response': {
                'type': 'error',
                'response': get_error_response(language, str(e))
            }
        }

def handle_follow_up_response(selected_option, conversation_context, language):
    """Handle follow-up response after user selects clarification option."""
    try:
        original_query = conversation_context.get('original_query', '')
        
        # Create more specific query by combining original with selected option
        specific_query = f"{original_query} - {selected_option}"
        
        # Retrieve relevant documents with the specific query
        results = hybrid_retriever.search(specific_query, TOP_K_RETRIEVAL, language)
        
        # Generate specific response using the enhanced client
        response = openai_client.handle_follow_up_response(
            original_query=original_query,
            selected_option=selected_option,
            context=results,
            language=language
        )
        
        # Format as direct response
        return {
            'success': True,
            'response': {
                'type': 'direct_answer',
                'response': response.response,
                'context': {
                    'awaiting_clarification': False
                },
                'metadata': {
                    'original_query': original_query,
                    'selected_option': selected_option,
                    'documents_found': len(results)
                }
            }
        }
        
    except Exception as e:
        logging.error(f"Error in handle_follow_up_response: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'response': {
                'type': 'error',
                'response': get_error_response(language, str(e))
            }
        }

def process_single_question_enhanced(query, language):
    """Process single question with enhanced vague detection."""
    try:
        # Retrieve relevant documents first
        results = hybrid_retriever.search(query, TOP_K_RETRIEVAL, language)
        
        # Generate response using enhanced OpenAI client
        response = openai_client.generate_response(
            query=query,
            context=results,
            language=language
        )
        
        # Handle different response types
        if hasattr(response, 'response_type'):
            if response.response_type == ResponseType.CLARIFICATION_NEEDED:
                # Return clarification request
                return {
                    'success': True,
                    'response': {
                        'type': 'clarification_needed',
                        'main_response': response.main_response,
                        'follow_up_question': response.follow_up_question,
                        'options': response.options,
                        'context': {
                            'original_query': query,
                            'awaiting_clarification': True
                        },
                        'metadata': {
                            'documents_found': len(results),
                            'query_type': 'vague'
                        }
                    }
                }
            else:
                # Direct response
                return {
                    'success': True,
                    'response': {
                        'type': 'direct_answer',
                        'response': response.response,
                        'context': {
                            'awaiting_clarification': False
                        },
                        'metadata': {
                            'documents_found': len(results),
                            'query_type': 'specific'
                        }
                    }
                }
        else:
            # Fallback for old response format
            return {
                'success': True,
                'response': {
                    'type': 'direct_answer',
                    'response': str(response),
                    'context': {
                        'awaiting_clarification': False
                    },
                    'metadata': {
                        'documents_found': len(results)
                    }
                }
            }
            
    except Exception as e:
        logging.error(f"Error in process_single_question_enhanced: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'response': {
                'type': 'error',
                'response': get_error_response(language, str(e))
            }
        }

def process_multiple_questions(questions, language):
    """Process multiple questions separately."""
    try:
        responses = []
        
        for question in questions:
            # Retrieve relevant documents for each question
            results = hybrid_retriever.search(question, TOP_K_RETRIEVAL, language)
            
            # Generate response for each question (force direct response for multi-question)
            if not results:
                answer = get_no_results_response(language)
            else:
                response = openai_client.generate_response(
                    query=question,
                    context=results,
                    language=language,
                    force_direct=True  # Skip vague detection for multi-question
                )
                answer = response.response if hasattr(response, 'response') else str(response)
                
            # Format individual response
            formatted_response = {
                'question': question,
                'answer': answer,
                'documents_found': len(results)
            }
            responses.append(formatted_response)
            
        # Combine responses
        combined_answer = format_multi_response(responses, language)
        
        return {
            'success': True,
            'response': {
                'type': 'direct_answer',
                'response': combined_answer,
                'context': {
                    'awaiting_clarification': False
                },
                'metadata': {
                    'question_count': len(questions),
                    'query_type': 'multiple'
                }
            }
        }
        
    except Exception as e:
        logging.error(f"Error in process_multiple_questions: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'response': {
                'type': 'error',
                'response': get_error_response(language, str(e))
            }
        }

def format_multi_response(responses, language):
    """Format multiple responses into a single answer."""
    if language == 'fr':
        formatted = "Voici les réponses à vos questions :\n\n"
        for i, resp in enumerate(responses, 1):
            formatted += f"**Question {i} :** {resp['question']}\n"
            formatted += f"**Réponse :** {resp['answer']}\n\n"
    else:  # Arabic
        formatted = "هاذي إجابات الأسئلة متاعك:\n\n"
        for i, resp in enumerate(responses, 1):
            formatted += f"**السؤال {i}:** {resp['question']}\n"
            formatted += f"**الإجابة:** {resp['answer']}\n\n"
    
    return formatted

def get_no_results_response(language):
    """Get response for when no relevant information is found."""
    if language == 'fr':
        return """
        Je n'ai pas trouvé d'informations spécifiques concernant votre question dans la documentation du RNE.
        Pourriez-vous reformuler votre question ou fournir plus de détails sur ce que vous recherchez?
        
        Vous pouvez également consulter directement le site officiel du Registre National des Entreprises (RNE) à l'adresse : https://www.registre-entreprises.tn/
        """
    else:
        return """
        ما لقيتش معلومات واضحة على سؤالك في وثائق السجل الوطني للمؤسسات.
        تنجم تعاود تطرح السؤال بطريقة أوضح؟ ولا تعطينا شوية تفاصيل زيادة؟

        تنجم زادة تدخل للموقع الرسمي متاع RNE من هنا: https://www.registre-entreprises.tn/
        """

def get_error_response(language, error_msg):
    """Get error response in the appropriate language."""
    if language == 'fr':
        return f"Désolé, une erreur s'est produite lors du traitement de votre demande. Erreur: {error_msg}"
    else:
        return f"آسف، صار خطأ وقت معالجة طلبك. الخطأ: {error_msg}"

# Legacy endpoint for backward compatibility
@app.route('/chat', methods=['POST'])
def legacy_chat():
    """Legacy chat endpoint that redirects to the new enhanced endpoint."""
    return chat()

if __name__ == '__main__':
    # Initialize components before starting the server
    try:
        initialize_components()
        print(f"Starting server on {HOST}:{PORT} (Debug: {DEBUG})")
        app.run(host=HOST, port=PORT, debug=DEBUG)
    except Exception as e:
        print(f"Failed to start server: {str(e)}")
        print(traceback.format_exc())