"""
Memory Consolidation - Merge similar facts, promote stable bubbles to semantic.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
from sqlalchemy.orm import Session

from contextmemory.db.models.memory import Memory
from contextmemory.memory.embeddings import embed_text
from contextmemory.memory.vector_store import get_vector_store, save_vector_store
from contextmemory.core.openai_client import get_llm_client
from contextmemory.core.settings import get_settings
import json
import re


SIMILARITY_THRESHOLD_MERGE = 0.85
SIMILARITY_THRESHOLD_PROMOTE = 0.75
BUBBLE_AGE_DAYS_FOR_PROMOTION = 7
MIN_BUBBLE_IMPORTANCE_FOR_PROMOTION = 0.6
MAX_BUBBLES_TO_PROMOTE_PER_RUN = 5


def consolidate_memories(db: Session, conversation_id: int) -> Dict:
    """
    Run memory consolidation for a conversation.
    
    Returns:
        Dict with stats: {"merged": int, "promoted": int, "details": [...]}
    """
    settings = get_settings()
    results = {"merged": 0, "promoted": 0, "details": []}
    
    # 1. Merge similar semantic facts
    merged_count = _merge_similar_semantic_facts(db, conversation_id)
    results["merged"] = merged_count
    if merged_count > 0:
        results["details"].append(f"Merged {merged_count} similar semantic facts")
    
    # 2. Promote stable bubbles to semantic
    promoted_count = _promote_bubbles_to_semantic(db, conversation_id)
    results["promoted"] = promoted_count
    if promoted_count > 0:
        results["details"].append(f"Promoted {promoted_count} bubbles to semantic facts")
    
    if settings.debug:
        print(f"[CONSOLIDATION] {results}")
    
    return results


def _merge_similar_semantic_facts(db: Session, conversation_id: int) -> int:
    """Find and merge similar semantic facts using LLM."""
    settings = get_settings()
    # Get all active semantic memories
    semantic_memories = db.query(Memory).filter(
        Memory.conversation_id == conversation_id,
        Memory.is_active == True,
        Memory.is_episodic == False
    ).all()
    
    if len(semantic_memories) < 2:
        return 0
    
    # Use FAISS to find similar pairs
    vector_store = get_vector_store(conversation_id)
    if vector_store.count == 0:
        return 0
    
    merged_count = 0
    processed_ids = set()
    
    for memory in semantic_memories:
        if memory.id in processed_ids:
            continue
        
        if not memory.embedding:
            continue
            
        # Search for similar memories
        results = vector_store.search(memory.embedding, k=10)
        
        for r in results:
            other_id = r["memory_id"]
            score = r["score"]
            
            if other_id == memory.id:
                continue
            if other_id in processed_ids:
                continue
            if score < SIMILARITY_THRESHOLD_MERGE:
                continue
            
            other_memory = db.get(Memory, other_id)
            if not other_memory or other_memory.is_episodic or not other_memory.is_active:
                continue
            
            # Ask LLM if they should be merged
            decision = _llm_decide_merge(memory.memory_text, other_memory.memory_text)
            
            if decision == "MERGE":
                # Merge: keep the more detailed/recent one, delete the other
                merged_text = _llm_merge_texts(memory.memory_text, other_memory.memory_text)
                
                # Update the kept memory
                memory.memory_text = merged_text
                memory.embedding = embed_text(merged_text)
                memory.updated_at = datetime.now(timezone.utc)
                vector_store.add(memory.id, memory.embedding)
                
                # Soft delete the other
                vector_store.remove(other_memory.id)
                other_memory.is_active = False
                
                processed_ids.add(other_id)
                merged_count += 1
                
                if settings.debug:
                    print(f"[CONSOLIDATION] Merged memory {other_id} into {memory.id}")
    
    if merged_count > 0:
        save_vector_store(conversation_id)
        db.commit()
    
    return merged_count


def _promote_bubbles_to_semantic(db: Session, conversation_id: int) -> int:
    """Promote old, stable, important bubbles to semantic facts."""
    settings = get_settings()
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=BUBBLE_AGE_DAYS_FOR_PROMOTION)
    
    # Get old, important, active bubbles
    bubbles = db.query(Memory).filter(
        Memory.conversation_id == conversation_id,
        Memory.is_active == True,
        Memory.is_episodic == True,
        Memory.importance >= MIN_BUBBLE_IMPORTANCE_FOR_PROMOTION,
        Memory.occurred_at <= cutoff_date
    ).order_by(Memory.importance.desc()).limit(MAX_BUBBLES_TO_PROMOTE_PER_RUN).all()
    
    if not bubbles:
        return 0
    
    promoted_count = 0
    vector_store = get_vector_store(conversation_id)
    
    for bubble in bubbles:
        # Check if bubble content is stable (not time-sensitive anymore)
        decision = _llm_decide_promotion(bubble.memory_text)
        
        if decision == "PROMOTE":
            # Convert to semantic
            bubble.is_episodic = False
            bubble.occurred_at = None
            bubble.session_id = None
            bubble.updated_at = datetime.now(timezone.utc)
            
            # Re-embed with semantic context
            bubble.embedding = embed_text(bubble.memory_text)
            vector_store.add(bubble.id, bubble.embedding)
            
            promoted_count += 1
            
            if settings.debug:
                print(f"[CONSOLIDATION] Promoted bubble {bubble.id} to semantic")
    
    if promoted_count > 0:
        save_vector_store(conversation_id)
        db.commit()
    
    return promoted_count


def _llm_decide_merge(text1: str, text2: str) -> str:
    """Ask LLM if two semantic facts should be merged."""
    settings = get_settings()
    client = get_llm_client()
    
    prompt = f"""Two semantic memories are very similar. Should they be merged?

Memory A: {text1}
Memory B: {text2}

Rules:
- MERGE if they convey the SAME fact (e.g., "User likes Python" + "User prefers Python")
- MERGE if one is a subset of the other (e.g., "User works at Google" + "User is a software engineer at Google")
- KEEP_SEPARATE if they are distinct facts about different things

Return ONLY: "MERGE" or "KEEP_SEPARATE\""""
    
    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": "You are a memory consolidation agent. Return only MERGE or KEEP_SEPARATE."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        result = response.choices[0].message.content.strip().upper()
        return result if result in ("MERGE", "KEEP_SEPARATE") else "KEEP_SEPARATE"
    except Exception as e:
        if settings.debug:
            print(f"[CONSOLIDATION] LLM merge decision error: {e}")
        return "KEEP_SEPARATE"


def _llm_merge_texts(text1: str, text2: str) -> str:
    """Ask LLM to merge two similar texts into one canonical version."""
    settings = get_settings()
    client = get_llm_client()
    
    prompt = f"""Merge these two similar facts into ONE canonical fact. Keep the most specific/complete information.

Memory A: {text1}
Memory B: {text2}

Return ONLY the merged fact (starting with "User ")."""
    
    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": "Merge two similar facts into one. Return only the merged fact."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        if settings.debug:
            print(f"[CONSOLIDATION] LLM merge error: {e}")
        return text1  # Fallback to first


def _llm_decide_promotion(bubble_text: str) -> str:
    """Ask LLM if a bubble should be promoted to semantic fact."""
    settings = get_settings()
    client = get_llm_client()
    
    prompt = f"""An episodic bubble (time-bound memory) is old enough to consider promotion to semantic fact.

Bubble: {bubble_text}

Rules:
- PROMOTE if the content is NOW a stable fact (e.g., "User is debugging JWT issue" → "User works with JWT authentication" if they keep mentioning it)
- PROMOTE if it describes a lasting trait/preference discovered during that episode
- KEEP_AS_BUBBLE if it's still time-sensitive (deadlines, one-time events, active debugging sessions)
- KEEP_AS_BUBBLE if it's a specific moment that shouldn't become a general fact

Return ONLY: "PROMOTE" or "KEEP_AS_BUBBLE\""""
    
    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": "Decide if an old bubble should become a semantic fact. Return only PROMOTE or KEEP_AS_BUBBLE."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        result = response.choices[0].message.content.strip().upper()
        return result if result in ("PROMOTE", "KEEP_AS_BUBBLE") else "KEEP_AS_BUBBLE"
    except Exception as e:
        if settings.debug:
            print(f"[CONSOLIDATION] LLM promotion error: {e}")
        return "KEEP_AS_BUBBLE"


def get_consolidation_stats(db: Session, conversation_id: int) -> Dict:
    """Get stats about consolidation opportunities."""
    semantic_count = db.query(Memory).filter(
        Memory.conversation_id == conversation_id,
        Memory.is_active == True,
        Memory.is_episodic == False
    ).count()
    
    bubble_count = db.query(Memory).filter(
        Memory.conversation_id == conversation_id,
        Memory.is_active == True,
        Memory.is_episodic == True
    ).count()
    
    old_bubbles = db.query(Memory).filter(
        Memory.conversation_id == conversation_id,
        Memory.is_active == True,
        Memory.is_episodic == True,
        Memory.importance >= MIN_BUBBLE_IMPORTANCE_FOR_PROMOTION,
        Memory.occurred_at <= datetime.now(timezone.utc) - timedelta(days=BUBBLE_AGE_DAYS_FOR_PROMOTION)
    ).count()
    
    return {
        "semantic_facts": semantic_count,
        "bubbles": bubble_count,
        "promotable_bubbles": old_bubbles,
        "merge_threshold": SIMILARITY_THRESHOLD_MERGE,
        "promotion_age_days": BUBBLE_AGE_DAYS_FOR_PROMOTION,
        "min_importance_for_promotion": MIN_BUBBLE_IMPORTANCE_FOR_PROMOTION
    }