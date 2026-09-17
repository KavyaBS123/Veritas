"""
ContextMemory Example - Interactive Chat with Memory

This example demonstrates how to use ContextMemory to build
a chatbot that remembers facts from conversations.
"""

from contextmemory import configure, Memory, create_table, SessionLocal
from contextmemory.core.openai_client import get_openai_client
from contextmemory.db.models.conversation import Conversation

# ANSI color codes
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def chat_with_memory(user_message: str, memory: Memory, conversation_id: int) -> str:
    """
    Generate a response using memories from past conversations.
    """
    client = get_openai_client()
    
    # Search for relevant memories
    relevant_memories = memory.search(
        query=user_message,
        conversation_id=conversation_id,
        limit=5,
    )

    # Format memories for the prompt
    memories_text = "\n".join(
        f"- {m['memory']}" for m in relevant_memories["results"]
    ) or "No memories yet."

    # Generate response with memory context
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant with memory. "
                "Use the user's memories to personalize your responses.\n\n"
                f"User Memories:\n{memories_text}"
            ),
        },
        {"role": "user", "content": user_message},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )

    assistant_message = response.choices[0].message.content

    # Store this conversation in memory
    memory.add(
        messages=[
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ],
        conversation_id=conversation_id,
    )

    return assistant_message


def main():
    """
    Main entry point - Interactive chat loop.
    """
    print(f"{CYAN}{'=' * 50}{RESET}")
    print(f"{CYAN}{BOLD}ContextMemory Chat Demo{RESET}")
    print(f"{CYAN}{'=' * 50}{RESET}")
    print()

    # Step 1: Configure (reads from environment variables or use configure())
    # configure(openai_api_key="sk-...")  # Or set OPENAI_API_KEY env var
    
    # Step 2: Create database tables
    create_table()

    # Step 3: Create session and memory instance
    db = SessionLocal()

    try:
        # Get or create conversation
        conversation_id = input("Enter conversation ID (or press Enter for new): ").strip()
        
        if conversation_id and conversation_id.isdigit():
            conversation_id = int(conversation_id)
            conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
            if not conversation:
                conversation = Conversation(id=conversation_id)
                db.add(conversation)
                db.commit()
                print(f"Created new conversation: {conversation_id}")
            else:
                print(f"Resuming conversation: {conversation_id}")
        else:
            conversation = Conversation()
            db.add(conversation)
            db.commit()
            conversation_id = conversation.id
            print(f"Created new conversation: {conversation_id}")

        # Initialize memory
        memory = Memory(db)

        print()
        print(f"{CYAN}Chat started! Type 'exit' to quit, 'memories' to see stored memories.{RESET}")
        print(f"{CYAN}{'-' * 50}{RESET}")

        # Chat loop
        while True:
            user_input = input(f"\n{GREEN}{BOLD}You:{RESET} ").strip()

            if not user_input:
                continue

            if user_input.lower() == "exit":
                print(f"{YELLOW}Goodbye!{RESET}")
                break

            if user_input.lower() == "memories":
                # Show all memories for this conversation (direct DB query, no embedding)
                from contextmemory.db.models.memory import Memory as MemoryModel
                memories = memory.db.query(MemoryModel).filter(
                    MemoryModel.conversation_id == conversation_id,
                    MemoryModel.is_active == True
                ).order_by(MemoryModel.created_at.desc()).limit(20).all()
                
                print(f"\n{CYAN}--- Stored Memories ---{RESET}")
                for m in memories:
                    mtype = "bubble" if m.is_episodic else "semantic"
                    color = BLUE if m.is_episodic else GREEN
                    print(f"  {color}[{m.id}] [{mtype}]{RESET} {m.memory_text}")
                if not memories:
                    print(f"  {YELLOW}No memories stored yet.{RESET}")
                print(f"{CYAN}{'-' * 25}{RESET}")
                continue

            if user_input.lower() == "consolidate":
                # Run memory consolidation
                print(f"{YELLOW}Running memory consolidation...{RESET}")
                result = memory.consolidate(conversation_id)
                print(f"{GREEN}Consolidation complete:{RESET} {result}")
                if result["details"]:
                    for detail in result["details"]:
                        print(f"  - {detail}")
                continue

            if user_input.lower() == "stats":
                # Show consolidation stats
                stats = memory.get_consolidation_stats(conversation_id)
                print(f"\n{CYAN}--- Consolidation Stats ---{RESET}")
                for k, v in stats.items():
                    print(f"  {k}: {v}")
                print(f"{CYAN}{'-' * 25}{RESET}")
                continue

            if user_input.lower() == "global":
                # Cross-conversation search
                query = input(f"{GREEN}Search query:{RESET} ").strip()
                if query:
                    exclude = input(f"{GREEN}Exclude current conversation? (y/N):{RESET} ").strip().lower() == 'y'
                    print(f"{YELLOW}Searching across all conversations...{RESET}")
                    result = memory.search_global(query, limit=10, exclude_conversation_id=conversation_id if exclude else None)
                    print(f"\n{CYAN}--- Global Search Results ---{RESET}")
                    for r in result["results"]:
                        conv_id = r.get("conversation_id", "?")
                        mtype = "bubble" if r["type"] == "bubble" else "semantic"
                        color = BLUE if r["type"] == "bubble" else GREEN
                        print(f"  {color}[{r['memory_id']}] [conv:{conv_id}] [{mtype}]{RESET} {r['memory'][:80]}... (score: {r['score']})")
                    if not result["results"]:
                        print(f"  {YELLOW}No results found.{RESET}")
                    print(f"{CYAN}{'-' * 30}{RESET}")
                continue

            if user_input.lower() == "rebuild-global":
                # Rebuild global index from DB
                print(f"{YELLOW}Rebuilding global index from database...{RESET}")
                from contextmemory.memory.vector_store import rebuild_global_index_from_db
                rebuild_global_index_from_db(db)
                print(f"{GREEN}Global index rebuilt.{RESET}")
                continue

            # Get AI response
            response = chat_with_memory(user_input, memory, conversation_id)
            print(f"\n{BLUE}{BOLD}AI:{RESET} {response}")

    finally:
        db.close()


if __name__ == "__main__":
    main()
