from typing import List
from core.openai_client import get_openai_client

client = get_openai_client()
EMBEDDING_MODEL = "text-embedding-3-small"

def embed_text(text: str) -> List[float]:
    """
    generate the embeddings of any text
    """
    
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding