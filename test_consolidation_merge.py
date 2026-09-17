#!/usr/bin/env python3
"""
Test merge functionality by adding similar facts.
"""
import sys
sys.path.insert(0, '/Users/kavya/Desktop/context_memory/context-memory/src')

from contextmemory.db.database import SessionLocal
from contextmemory.memory import ContextMemory

def main():
    db = SessionLocal()
    memory = ContextMemory(db)
    conversation_id = 2  # Use existing conversation with memories
    
    print("Adding similar/duplicate facts to test merge...")
    
    # Add facts that are similar to existing ones
    test_messages = [
        {"role": "user", "content": "My name is Kavya and I work as a backend engineer"},
        {"role": "assistant", "content": "Nice to meet you Kavya!"},
    ]
    
    result = memory.add(test_messages, conversation_id)
    print(f"Added: {result}")
    
    # Add another similar fact
    test_messages2 = [
        {"role": "user", "content": "I work with FastAPI for building APIs"},
        {"role": "assistant", "content": "FastAPI is great for APIs!"},
    ]
    
    result2 = memory.add(test_messages2, conversation_id)
    print(f"Added: {result2}")
    
    # Now run consolidation
    print("\nRunning consolidation...")
    result = memory.consolidate(conversation_id)
    print(f"Result: {result}")
    
    # Show all memories
    from contextmemory.db.models.memory import Memory
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