#!/usr/bin/env python3
"""
Test script for memory consolidation.
"""
import sys
sys.path.insert(0, '/Users/kavya/Desktop/context_memory/context-memory/src')

from contextmemory.db.database import SessionLocal
from contextmemory.memory import ContextMemory, get_consolidation_stats

def main():
    db = SessionLocal()
    memory = ContextMemory(db)
    
    # Check all conversations
    from contextmemory.db.models.conversation import Conversation
    convs = db.query(Conversation).all()
     
    for conv in convs:
        print(f"\n{'='*60}")
        print(f"Conversation {conv.id} (created: {conv.created_at})")
        print(f"{'='*60}")
        
        # Get consolidation stats
        stats = memory.get_consolidation_stats(conv.id)
        print(f"Stats: {stats}")
        
        # Show memories
        from contextmemory.db.models.memory import Memory
        mems = db.query(Memory).filter(
            Memory.conversation_id == conv.id,
            Memory.is_active == True
        ).all()
        
        print(f"\nMemories ({len(mems)} total):")
        for m in mems:
            mtype = "BUBBLE" if m.is_episodic else "SEMANTIC"
            print(f"  [{m.id}] {mtype} (importance={m.importance:.2f}) {m.memory_text[:80]}...")
            if m.occurred_at:
                print(f"       occurred: {m.occurred_at}")
        
        # Run consolidation
        if stats["semantic_facts"] > 1 or stats["promotable_bubbles"] > 0:
            print(f"\nRunning consolidation...")
            result = memory.consolidate(conv.id)
            print(f"Result: {result}")
        else:
            print(f"\nSkipping consolidation (not enough data)")

    db.close()

if __name__ == "__main__":
    main()