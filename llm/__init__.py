"""
LLM package for RNE Chatbot.
"""

from llm.openai_client import OpenAIClient
from llm.prompt_templates import (
    SYSTEM_PROMPT_FR,
    SYSTEM_PROMPT_AR,
    QUESTION_SEGMENTATION_PROMPT,
    format_context,
    get_no_results_response,
    format_final_response
)

__all__ = [
    'OpenAIClient',
    'SYSTEM_PROMPT_FR',
    'SYSTEM_PROMPT_AR',
    'QUESTION_SEGMENTATION_PROMPT',
    'format_context',
    'get_no_results_response',
    'format_final_response'
]