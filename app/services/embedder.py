import google.generativeai as genai
from app.config import settings

genai.configure(api_key=settings.gemini_api_key)

# We lock this to 768 to match your new Qdrant collection
EMBEDDING_DIMENSION: int = 768  
_BATCH_SIZE: int = 100 # Reduced batch size slightly for stability

async def embed_query(text: str) -> list[float]:
    """
    Generate a single embedding vector for a query string.
    """
    result = await genai.embed_content_async(
        model=settings.gemini_embedding_model,
        content=text,
        task_type="retrieval_query",
        output_dimensionality=EMBEDDING_DIMENSION
    )
    return result['embedding']

async def embed_chunks(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of text chunks.
    """
    if not texts:
        return []

    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        
        result = await genai.embed_content_async(
            model=settings.gemini_embedding_model,
            content=batch,
            task_type="retrieval_document",
            output_dimensionality=EMBEDDING_DIMENSION
        )
        

        batch_embeddings = result['embedding']
        
        all_embeddings.extend(batch_embeddings)

    return all_embeddings