"""
LLM package for RNE Chatbot.
"""

from llm.groq_client import GroqClient
from llm.prompt_templates import (
    SYSTEM_PROMPT_FR,
    SYSTEM_PROMPT_AR,
    QUESTION_SEGMENTATION_PROMPT,
    format_context,
    get_no_results_response,
    format_final_response
)

__all__ = [
    'GroqClient',
    'SYSTEM_PROMPT_FR',
    'SYSTEM_PROMPT_AR',
    'QUESTION_SEGMENTATION_PROMPT',
    'format_context',
    'get_no_results_response',
    'format_final_response'
]