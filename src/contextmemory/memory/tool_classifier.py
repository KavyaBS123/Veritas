"""
Tool Classifier - Decides what action to take with a candidate fact.

Uses JSON-based output for compatibility with all LLM providers (OpenAI, Claude, etc).
"""

import json
import re
from typing import List, Optional
from dataclasses import dataclass

from contextmemory.core.openai_client import get_llm_client
from contextmemory.core.settings import get_settings
from contextmemory.utils.tool_call_system_prompt import TOOL_CALL_SYSTEM_PROMPT


@dataclass
class ToolDecision:
    """
    Result of tool classification.
    
    Attributes:
        action: One of "ADD", "UPDATE", "DELETE", "NOOP"
        memory_id: ID of memory to update/delete (None for ADD/NOOP)
        text: Text to store for ADD/UPDATE (None for DELETE/NOOP)
    """
    action: str
    memory_id: Optional[int]
    text: Optional[str]


def llm_tool_call(candidate_fact: str, similar_memories: List) -> ToolDecision:
    """
    LLM decides which action to take with a candidate fact.
    
    Args:
        candidate_fact: The fact to potentially store
        similar_memories: List of existing Memory objects that are similar
        
    Returns:
        ToolDecision with action, memory_id, and text
    """
    settings = get_settings()
    client = get_llm_client()
    
    # Format existing memories for context
    if similar_memories:
        memory_context = "\n".join(
            f"- ID {m.id}: {m.memory_text}" for m in similar_memories
        )
    else:
        memory_context = "No existing memories found."

    # Build messages
    messages = [
        {"role": "system", "content": TOOL_CALL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""Candidate fact:
{candidate_fact}

Existing similar memories:
{memory_context}

Decide the action."""
        }
    ]

    # Call LLM
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0
    )

    raw_output = response.choices[0].message.content
    
    if settings.debug:
        print(f"[DEBUG] Tool classifier output: {raw_output}")
    
    # Parse JSON response
    return _parse_decision(raw_output, candidate_fact, settings.debug)


def _parse_decision(raw_output: str, candidate_fact: str, debug: bool = False) -> ToolDecision:
    """
    Parse the LLM's JSON response into a ToolDecision.
    
    Handles markdown code blocks and provides fallback behavior.
    """
    try:
        json_str = raw_output
        
        # Extract JSON from markdown code blocks if present
        if "```" in json_str:
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', json_str)
            if match:
                json_str = match.group(1)
        
        json_str = json_str.strip()
        result = json.loads(json_str)
        
        # Normalize action to uppercase
        action = result.get("action", "NOOP").upper()
        
        # Validate action
        if action not in ("ADD", "UPDATE", "DELETE", "REPLACE", "NOOP"):
            action = "ADD"  # Default to ADD for unknown actions
        
        return ToolDecision(
            action=action,
            memory_id=result.get("memory_id"),
            text=result.get("text", candidate_fact)
        )
        
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        if debug:
            print(f"[DEBUG] Tool classifier parse error: {e}")
            print(f"[DEBUG] Raw output was: {raw_output[:200]}...")
        
        # Default to ADD if parsing fails - better to store than miss
        return ToolDecision(
            action="ADD",
            memory_id=None,
            text=candidate_fact
        )
