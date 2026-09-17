#!/usr/bin/env python3
"""
Test bubble promotion with a bubble that should become semantic.
"""
import sys
sys.path.insert(0, '/Users/kavya/Desktop/context_memory/context-memory/src')

from contextmemory.db.database import SessionLocal
from contextmemory.memory import ContextMemory
from contextmemory.db.models.memory import Memory
from contextmemory.memory.embeddings import embed_text
from contextmemory.memory.vector_store import get_vector_store, save_vector_store
from datetime import datetime, timedelta, timezone

def main():
    db = SessionLocal()
    memory = ContextMemory(db)
    conversation_id = 2
    
    # Add a bubble that represents a stable preference/trait (should be promotable)
    vector_store = get_vector_store(conversation_id)
    
    bubble_text = "User prefers dark mode for all applications"
    embedding = embed_text(bubble_text)
    
    bubble = Memory(
        conversation_id=conversation_id,
        memory_text=bubble_text,
        embedding=embedding,
        is_episodic=True,
        occurred_at=datetime.now(timezone.utc) - timedelta(days=10),  # Old
        importance=0.8,
        is_active=True,
        memory_metadata={}
    )
    
    db.add(bubble)
    db.flush()
    vector_store.add(bubble.id, embedding)
    save_vector_store(conversation_id)
    db.commit()
    
    print(f"Added test bubble {bubble.id}: {bubble_text}")
    
    # Test promotion with lowered threshold
    import contextmemory.memory.consolidation as consolidation_module
    original_threshold = consolidation_module.BUBBLE_AGE_DAYS_FOR_PROMOTION
    consolidation_module.BUBBLE_AGE_DAYS_FOR_PROMOTION = 1
    
    print(f"\nRunning consolidation...")
    result = memory.consolidate(conversation_id)
    print(f"Result: {result}")
    
    # Restore
    consolidation_module.BUBBLE_AGE_DAYS_FOR_PROMOTION = original_threshold
    
    # Show all memories
    mems = db.query(Memory).filter(
        Memory.conversation_id == conversation_id,
        Memory.is_active == True
    ).all()
    
    print(f"\nAll memories after consolidation:")
    for m in mems:
        mtype = "BUBBLE" if m.is_episodic else "SEMANTIC"
        print(f"  [{m.id}] {mtype} {m.memory_text[:100]}")
    
    db.close()

if __name__ == "__main__":
    main()