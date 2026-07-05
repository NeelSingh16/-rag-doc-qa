import chromadb
import os

CHROMA_DB_PATH = "./chroma_db"

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space" : "cosine"}
)

def store_chunks(filename: str, chunks: list[dict], embeddings: list[list[float]]):
    ids = [f"{filename}_chunk_{chunk['chunk_index']}" for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]

    metadatas = [
        {
            "filename": filename,
            "chunk_index": chunk["chunk_index"],
            "token_count": chunk["token_count"]
        }
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    return len(ids)

def query_similar_chunks(query_embedding: list[float], filename: str, top_k: int=5) -> list[dict]:

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"filename": filename},
        include=["documents", "metadatas", "distances"]
    )

    chunks=[]
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "similarity_score": 1 - results["distances"][0][i]
        })

    return chunks