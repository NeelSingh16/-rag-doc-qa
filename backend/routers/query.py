from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.embedding import get_embedding
from services.vector_store import query_similar_chunks
from services.llm import get_answer

router = APIRouter()


# Pydantic model defines the shape of the request body
# FastAPI uses this to automatically validate incoming JSON
class QueryRequest(BaseModel):
    filename: str   # which document to search in
    question: str


@router.post("/query")
def query_document(request: QueryRequest):
    """
    Full RAG query pipeline:
    1. Embed the user's question
    2. Find the most similar chunks in ChromaDB
    3. Send chunks + question to the LLM
    4. Return the answer
    """

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not request.filename.strip():
        raise HTTPException(status_code=400, detail="Filename cannot be empty.")

    # Step 1: Embed the question using the same model we used for chunks
    # This puts the question in the same vector space as our stored chunks
    question_embedding = get_embedding(request.question)

    # Step 2: Find top 5 most similar chunks from ChromaDB
    # These are the chunks most likely to contain the answer
    relevant_chunks = query_similar_chunks(
        query_embedding=question_embedding,
        filename=request.filename,
        top_k=5
    )

    if not relevant_chunks:
        raise HTTPException(
            status_code=404,
            detail="No chunks found for this document. Make sure you processed it first via /process/{filename}."
        )

    # Step 3: Send question + relevant chunks to the LLM and get answer
    result = get_answer(request.question, relevant_chunks)

    # Step 4: Return answer along with the chunks used (useful for debugging + citations)
    return {
        "question": request.question,
        "answer": result["answer"],
        "model_used": result["model_used"],
        "chunks_used": result["chunks_used"],
        "source_chunks": [
            {
                "text": chunk["text"][:200] + "...",  # truncate for readability
                "similarity_score": round(chunk["similarity_score"], 3),
                "chunk_index": chunk["metadata"]["chunk_index"]
            }
            for chunk in relevant_chunks
        ]
    }