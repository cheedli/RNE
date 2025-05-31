"""
Main Flask application for RNE Chatbot.
"""

import os
import json
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
from llm.openai_client import OpenAIClient
from utils.language_detector import LanguageDetector
from utils.response_formatter import ResponseFormatter

# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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
    
    # Initialize OpenAI client
    openai_client = OpenAIClient()
    
    print("All components initialized successfully!")

@app.route('/')
def index():
    """Render the main page of the application."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    API endpoint for chatting with the RNE chatbot.
    
    Expects JSON with:
    - message: User query text
    - language: (Optional) Preferred language ('fr' or 'ar')
    
    Returns:
    - JSON response with generated answer and metadata
    """
    try:
        data = request.json
        user_query = data.get('message', '')
        preferred_language = data.get('language')
        
        # Validate input
        if not user_query:
            return jsonify({'error': 'Missing message parameter'}), 400
            
        # Detect language if not specified
        if not preferred_language:
            preferred_language = language_detector.detect_language(user_query)
            
        # Process query
        return process_query(user_query, preferred_language)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_query(query, language):
    """
    Process a user query through the chatbot pipeline.
    
    Args:
        query: User query text.
        language: Language code ('fr' or 'ar').
        
    Returns:
        JSON response with generated answer and metadata.
    """
    # Check if query contains multiple questions
    questions = openai_client.segment_questions(query)
    
    # If there's only one question, process it directly
    if len(questions) == 1:
        # Retrieve relevant documents
        results = hybrid_retriever.search(query, TOP_K_RETRIEVAL, language)
        
        # If no relevant documents found
        if not results:
            response = get_no_results_response(language)
            return jsonify(ResponseFormatter.format_response(response, query, [], language))
            
        # Generate response
        answer = openai_client.generate_response(query, results, language)
        
        # Format response
        formatted_response = ResponseFormatter.format_response(answer, query, results, language)
        return jsonify(formatted_response)
    
    # Process multiple questions
    else:
        responses = []
        
        for question in questions:
            # Retrieve relevant documents for each question
            results = hybrid_retriever.search(question, TOP_K_RETRIEVAL, language)
            
            # Generate response for each question
            if not results:
                answer = get_no_results_response(language)
            else:
                answer = openai_client.generate_response(question, results, language)
                
            # Format individual response
            formatted_response = ResponseFormatter.format_response(answer, question, results, language)
            responses.append(formatted_response)
            
        # Combine responses
        combined_response = ResponseFormatter.format_multi_response(responses, query, language)
        return jsonify(combined_response)

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


if __name__ == '__main__':
    # Initialize components before starting the server
    initialize_components()
    
    # Start the Flask server
    app.run(host='0.0.0.0', port=PORT, debug=DEBUG)
