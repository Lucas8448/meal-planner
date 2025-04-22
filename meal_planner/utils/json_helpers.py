"""JSON helper utilities for cleaning and parsing LLM outputs."""

import json
import re
from typing import Any, Dict, List, Tuple, Union, Optional


def clean_markdown_code_block(text: str) -> str:
    """Clean markdown code blocks from LLM output.
    
    Args:
        text: Raw text from LLM that may contain markdown code blocks.
        
    Returns:
        Cleaned content without markdown formatting.
    """
    # Strip whitespace
    clean_text = text.strip()
    
    # Remove ```json and ```
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:].strip()
    elif clean_text.startswith("```"):
        clean_text = clean_text[3:].strip()
        
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3].strip()
        
    return clean_text


def parse_llm_json_output(llm_output: str) -> Tuple[Any, Optional[str]]:
    """Parse JSON output from an LLM, handling markdown and errors gracefully.
    
    Args:
        llm_output: Raw output from LLM.
        
    Returns:
        Tuple of (parsed_data, error_message). If successful, error_message is None.
    """
    clean_output = clean_markdown_code_block(llm_output)
    
    try:
        parsed_data = json.loads(clean_output)
        return parsed_data, None
    except json.JSONDecodeError as e:
        error_msg = f"JSON parse error: {e}"
        print(f"ERROR (parse_llm_json_output): {error_msg}")
        print(f"Raw output was: {llm_output}")
        return None, error_msg


def validate_list_data(data: Any, expected_type: type) -> Tuple[List, Optional[str]]:
    """Validate that data is a list of the expected type.
    
    Args:
        data: Data to validate
        expected_type: Expected type for list items
        
    Returns:
        Tuple of (valid_list, error_message). If successful, error_message is None.
    """
    if not isinstance(data, list):
        return [], f"Expected list but got {type(data).__name__}"
    
    valid_items = []
    for i, item in enumerate(data):
        if isinstance(item, expected_type):
            valid_items.append(item)
        else:
            print(f"WARNING: Item at index {i} has unexpected type {type(item).__name__}. Expected {expected_type.__name__}")
    
    return valid_items, None if valid_items else "No valid items found in list"


def validate_dict_keys(data: Dict, required_keys: List[str]) -> Tuple[bool, Optional[str]]:
    """Validate that a dictionary contains all required keys.
    
    Args:
        data: Dictionary to validate
        required_keys: List of required keys
        
    Returns:
        Tuple of (is_valid, error_message). If successful, error_message is None.
    """
    if not isinstance(data, dict):
        return False, f"Expected dict but got {type(data).__name__}"
    
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        return False, f"Missing required keys: {', '.join(missing_keys)}"
    
    return True, None 