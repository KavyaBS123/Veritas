from typing import List
from contextmemory.core.openai_client import get_embedding_client
from contextmemory.core.settings import get_settings


def embed_text(text: str) -> List[float]:
    """
    Generate the embeddings of any text.
    
    Uses the configured provider (OpenAI or OpenRouter).
    For OpenRouter, uses the openai/text-embedding-3-small model format.
    """
    settings = get_settings()
    client = get_embedding_client()
    
    # OpenRouter requires the provider prefix for embedding models
    # Only add prefix if not already present
    model = settings.embedding_model
    if settings.llm_provider == "openrouter" and not model.startswith("openai/"):
        model = f"openai/{model}"
    
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding