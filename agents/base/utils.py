# agents/base/utils.py
"""
Utility functions for AI agents
"""

import json
import logging
import re
from typing import Any, Dict, Optional


def safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    """
    Safely parse JSON from AI response text
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        Parsed JSON object or None if parsing fails
    """
    try:
        # Try direct parsing first
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from code blocks
    json_patterns = [
        r'```json\n(.*?)\n```',
        r'```\n(.*?)\n```',
        r'\{.*?\}',
        r'\[.*?\]'
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
    
    logging.warning(f"Failed to parse JSON from text: {text[:200]}...")
    return None


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text (4 chars per token approximation)
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    return max(1, len(text) // 4)


def calculate_confidence(indicators: Dict[str, Any]) -> float:
    """
    Calculate confidence score based on various indicators
    
    Args:
        indicators: Dictionary of confidence indicators
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    base_score = 0.5
    
    # Adjust based on available indicators
    if indicators.get('has_contact_info'):
        base_score += 0.2
    
    if indicators.get('has_pricing_info'):
        base_score += 0.15
    
    if indicators.get('has_services_description'):
        base_score += 0.15
    
    if indicators.get('website_quality_score', 0) > 0.7:
        base_score += 0.1
    
    # Penalize red flags
    red_flags = indicators.get('red_flags', [])
    if red_flags:
        base_score -= len(red_flags) * 0.1
    
    return max(0.0, min(1.0, base_score))


def clean_extracted_text(text: str) -> str:
    """
    Clean and normalize extracted text
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\-\.\,\!\?\:\;]', '', text)
    
    return text.strip()