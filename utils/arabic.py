"""
Arabic language utilities for detection and translation.
"""
import re
from langchain_core.output_parsers import StrOutputParser
from models.llm import get_langchain_llm
from prompts.templates import arabic_prompt

def is_query_arabic(query: str) -> bool:
    """
    Detect if a query contains Arabic characters.
    Args:
        query: Input text to check
    Returns:
        bool: True if Arabic characters are detected, False otherwise
    """
    # Arabic Unicode range: U+0600 to U+06FF
    return bool(re.search(r'[\u0600-\u06FF]', query))

def create_arabic_chain():
    """
    Create a LangChain for translating English to Arabic.
    Returns:
        A runnable chain that translates text to Arabic
    """
    # Get LLM
    llm = get_langchain_llm()
    # Create and return the translation chain
    return arabic_prompt | llm | StrOutputParser()

def translate_to_arabic(text: str) -> str:
    """
    Translate text to Arabic.
    Args:
        text: English text to translate
    Returns:
        str: Arabic translation
    """
    chain = create_arabic_chain()
    return chain.invoke({"answer": text})