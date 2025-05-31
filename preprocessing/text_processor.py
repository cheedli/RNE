"""
Text preprocessing utilities for the RNE chatbot.
"""

import re
import nltk
from typing import List, Optional
from langdetect import detect

# Try to load stopwords, but don't fail if they're not available
try:
    from nltk.corpus import stopwords
    nltk.data.find('corpora/stopwords')
    NLTK_STOPWORDS_AVAILABLE = True
except LookupError:
    print("NLTK stopwords not found. Downloading...")
    nltk.download('stopwords')
    try:
        from nltk.corpus import stopwords
        NLTK_STOPWORDS_AVAILABLE = True
    except:
        print("Warning: Failed to load NLTK stopwords. Will use basic stopword filtering.")
        NLTK_STOPWORDS_AVAILABLE = False

class TextProcessor:
    """Class for text preprocessing operations."""
    
    def __init__(self):
        """Initialize the text processor."""
        # Define basic French stopwords if NLTK is not available
        self.basic_fr_stopwords = {
            'le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'de', 'du', 'à', 'au', 'aux',
            'ce', 'cette', 'ces', 'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses', 
            'que', 'qui', 'quoi', 'dont', 'où', 'quand', 'comment', 'pourquoi', 
            'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles', 'on',
            'pour', 'par', 'en', 'dans', 'sur', 'sous', 'avec', 'sans', 'chez', 'entre'
        }
        
        # Define basic Arabic stopwords if NLTK is not available
        self.basic_ar_stopwords = {
            'من', 'إلى', 'عن', 'على', 'في', 'هذا', 'هذه', 'هؤلاء', 'ذلك', 'تلك', 'أولئك',
            'الذي', 'التي', 'الذين', 'اللواتي', 'أنا', 'أنت', 'هو', 'هي', 'نحن', 'أنتم', 'هم', 'هن',
            'كان', 'كانت', 'كانوا', 'يكون', 'تكون', 'يكونوا', 'كن', 'أن', 'لأن', 'لكن', 'إذا', 'لو'
        }
        
        # Use NLTK stopwords if available
        if NLTK_STOPWORDS_AVAILABLE:
            try:
                self.fr_stopwords = set(stopwords.words('french'))
                self.ar_stopwords = set(stopwords.words('arabic'))
            except:
                print("Warning: Error loading NLTK stopwords. Using basic stopwords instead.")
                self.fr_stopwords = self.basic_fr_stopwords
                self.ar_stopwords = self.basic_ar_stopwords
        else:
            self.fr_stopwords = self.basic_fr_stopwords
            self.ar_stopwords = self.basic_ar_stopwords
        
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text.
        
        Args:
            text: Input text.
            
        Returns:
            Language code ('fr', 'ar', or 'en' if not detected).
        """
        try:
            lang = detect(text)
            if lang == 'fr':
                return 'fr'
            elif lang in ['ar', 'arb']:
                return 'ar'
            return lang
        except:
            # Default to French if detection fails
            return 'fr'
    
    def normalize_text(self, text: str, language: Optional[str] = None) -> str:
        """
        Normalize text by removing special characters and extra whitespace.
        
        Args:
            text: Input text to normalize.
            language: Language of the text ('fr' or 'ar'). If None, will be detected.
            
        Returns:
            Normalized text.
        """
        if not language:
            language = self.detect_language(text)
            
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove special characters while preserving Arabic characters if needed
        if language == 'ar':
            # Preserve Arabic unicode range while removing special chars
            text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
        else:
            # For non-Arabic, remove special chars but keep alphanumeric and accented chars
            text = re.sub(r'[^\w\s\u00C0-\u017F]', ' ', text)
            
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str, language: Optional[str] = None) -> List[str]:
        """
        Tokenize text into words using a simple space-based approach.
        
        Args:
            text: Input text to tokenize.
            language: Language of the text ('fr' or 'ar'). If None, will be detected.
            
        Returns:
            List of tokens.
        """
        if not language:
            language = self.detect_language(text)
            
        # Simple tokenization by whitespace
        tokens = text.split()
        
        return tokens
    
    def remove_stopwords(self, tokens: List[str], language: Optional[str] = None) -> List[str]:
        """
        Remove stopwords from a list of tokens.
        
        Args:
            tokens: List of tokens.
            language: Language of the tokens ('fr' or 'ar'). If None, will be detected.
            
        Returns:
            List of tokens with stopwords removed.
        """
        if not tokens:
            return []
            
        if not language:
            # Try to detect language from the joined tokens
            language = self.detect_language(' '.join(tokens))
            
        if language == 'fr':
            return [token for token in tokens if token.lower() not in self.fr_stopwords]
        elif language == 'ar':
            return [token for token in tokens if token not in self.ar_stopwords]
        else:
            # For other languages, don't remove stopwords
            return tokens
    
    def preprocess(self, text: str, language: Optional[str] = None) -> List[str]:
        """
        Full preprocessing pipeline: normalize, tokenize, and remove stopwords.
        
        Args:
            text: Input text to preprocess.
            language: Language of the text ('fr' or 'ar'). If None, will be detected.
            
        Returns:
            List of preprocessed tokens.
        """
        if not language:
            language = self.detect_language(text)
            
        normalized_text = self.normalize_text(text, language)
        tokens = self.tokenize(normalized_text, language)
        filtered_tokens = self.remove_stopwords(tokens, language)
        
        return filtered_tokens
    
    def segment_questions(self, text: str) -> List[str]:
        """
        Segment text into multiple questions if present.
        
        Args:
            text: Input text that may contain multiple questions.
            
        Returns:
            List of individual questions.
        """
        # Simple rule-based segmentation by question marks and new lines
        segments = re.split(r'[?؟\n]+', text)
        
        # Filter out empty segments and add back the question mark
        questions = [segment.strip() + '?' for segment in segments if segment.strip()]
        
        # If no questions were detected, return the original text as a single item
        if not questions:
            return [text]
            
        return questions