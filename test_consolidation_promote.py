#!/usr/bin/env python3
"""
Test bubble promotion by temporarily lowering age threshold.
"""
import sys
sys.path.insert(0, '/Users/kavya/Desktop/context_memory/context-memory/src')

from contextmemory.db.database import SessionLocal
from contextmemory.memory import ContextMemory
from contextmemory.db.models.memory import Memory
from datetime import datetime, timedelta, timezone

def main():
    db = SessionLocal()
    memory = ContextMemory(db)
    conversation_id = 2
    
    # First, let's manually make a bubble old enough for promotion
    # by updating its occurred_at to be > 7 days ago
    bubble = db.query(Memory).filter(
        Memory.conversation_id == conversation_id,
        Memory.is_episodic == True,
        Memory.is_active == True
    ).first()
    
    if bubble:
        print(f"Found bubble {bubble.id}: {bubble.memory_text}")
        print(f"Original occurred_at: {bubble.occurred_at}")
        
        # Make it 10 days old
        bubble.occurred_at = datetime.now(timezone.utc) - timedelta(days=10)
        db.commit()
        print(f"Updated occurred_at: {bubble.occurred_at}")
    
    # Now test promotion with modified threshold
    # Temporarily monkey-patch the constant
    import contextmemory.memory.consolidation as consolidation_module
    original_threshold = consolidation_module.BUBBLE_AGE_DAYS_FOR_PROMOTION
    consolidation_module.BUBBLE_AGE_DAYS_FOR_PROMOTION = 1  # 1 day for testing
    
    print(f"\nRunning consolidation with lowered threshold...")
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