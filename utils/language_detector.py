"""
Language detection utilities for the RNE chatbot.
"""

import re
from langdetect import detect, LangDetectException

class LanguageDetector:
    """Class for language detection and related utilities."""
    
    def __init__(self, supported_languages=None, default_language='fr'):
        """
        Initialize the language detector.
        
        Args:
            supported_languages: List of supported language codes.
            default_language: Default language to use if detection fails.
        """
        self.supported_languages = supported_languages or ['fr', 'ar']
        self.default_language = default_language
        
        # Regular expressions for language detection backup
        self.arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
        
    def detect_language(self, text):
        """
        Detect the language of the input text.
        
        Args:
            text: Input text to detect language.
            
        Returns:
            Detected language code or default language if detection fails.
        """
        if not text or len(text.strip()) < 2:
            return self.default_language
        
        # First try using langdetect
        try:
            detected_lang = detect(text)
            
            # Map similar language codes
            if detected_lang in ['ar', 'arb']:
                return 'ar'
            elif detected_lang == 'fr':
                return 'fr'
                
            # If detected language is not supported, try regex patterns
            if detected_lang not in self.supported_languages:
                return self._detect_with_regex(text)
                
            return detected_lang
            
        except LangDetectException:
            # Fallback to regex pattern matching
            return self._detect_with_regex(text)
    
    def _detect_with_regex(self, text):
        """
        Detect language using regex patterns.
        
        Args:
            text: Input text.
            
        Returns:
            Detected language code or default language.
        """
        # Check for Arabic characters
        if self.arabic_pattern.search(text):
            return 'ar'
            
        # Default to French if no patterns match
        return self.default_language
    
    def is_arabic(self, text):
        """
        Check if text is in Arabic.
        
        Args:
            text: Input text.
            
        Returns:
            True if text is in Arabic, False otherwise.
        """
        return self.detect_language(text) == 'ar'
    
    def is_french(self, text):
        """
        Check if text is in French.
        
        Args:
            text: Input text.
            
        Returns:
            True if text is in French, False otherwise.
        """
        return self.detect_language(text) == 'fr'
    
    def get_direction(self, language):
        """
        Get text direction for the given language.
        
        Args:
            language: Language code.
            
        Returns:
            'rtl' for right-to-left languages, 'ltr' otherwise.
        """
        rtl_languages = ['ar']
        return 'rtl' if language in rtl_languages else 'ltr'