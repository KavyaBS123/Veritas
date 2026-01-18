import json
import re
from typing import List, Dict, Any
from contextmemory.core.openai_client import get_llm_client
from contextmemory.core.settings import get_settings
from contextmemory.utils.extraction_system_prompt import EXTRACTION_SYSTEM_PROMPT


def extract_memories(latest_pair: List[str], summary_text: str, recent_messages: List[str]) -> Dict[str, Any]:
    """
    Use LLM to extract candidate memory facts (semantic facts and bubbles).
    
    Returns:
        {
            "semantic": ["fact1", "fact2"],
            "bubbles": [{"text": "...", "importance": 0.7}]
        }
    """
    settings = get_settings()
    llm_client = get_llm_client()

    # List of string -> Single string
    recent_msgs_text = "\n".join(recent_messages)
    latest_pair_text = "\n".join(latest_pair)

    # Format msgs to give to LLM for extraction
    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {
            "role": "user", 
            "content" : f"""
Conversation Summary:
{summary_text}

Recent Messages:
{recent_msgs_text}

Latest Interaction:
{latest_pair_text}

Extract memory facts (semantic facts and bubbles).
"""
        }
    ]

    # Get model from settings (supports OpenRouter format like "openai/gpt-4o-mini")
    model = settings.llm_model

    # LLM extracts memory facts 
    response = llm_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1
    )

    raw_output = response.choices[0].message.content
    
    # Debug logging
    if settings.debug:
        print(f"[DEBUG] Raw LLM output: {raw_output[:500]}...")
    
    # Parse JSON - handle markdown code blocks
    try:
        # Try to extract JSON from markdown code blocks if present
        json_str = raw_output
        
        # Remove markdown code block formatting (```json ... ```)
        if "```" in json_str:
            # Find content between code blocks
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', json_str)
            if match:
                json_str = match.group(1)
        
        # Clean up any leading/trailing whitespace
        json_str = json_str.strip()
        
        result = json.loads(json_str)
        
        # Validate structure
        if "semantic" not in result:
            result["semantic"] = []
        if "bubbles" not in result:
            result["bubbles"] = []
            
        if settings.debug:
            print(f"[DEBUG] Extracted: {len(result['semantic'])} semantic, {len(result['bubbles'])} bubbles")
            
        return result
    except json.JSONDecodeError as e:
        if settings.debug:
            print(f"[DEBUG] JSON parse error: {e}")
            print(f"[DEBUG] Attempted to parse: {json_str[:200]}...")
        return {"semantic": [], "bubbles": []}