import os
from sentence_transformers import SentenceTransformer

# all-MiniLM-L6-v2 is a lightweight, fast model that produces
# 384-dimensional embeddings — good quality for RAG, downloads once (~80MB)
model = SentenceTransformer("all-MiniLM-L6-v2")

# load_dotenv()
#
# client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

# EMBEDDING_MODEL = "text-embedding-3-small"

def get_embedding(text: str) -> list[float]:
    embedding = model.encode(text)
    return embedding.tolist()

    # text = text.replace("\n", " ")
    # response = client.embeddings.create(
    #     input=text,
    #     model=EMBEDDING_MODEL
    # )
    #
    # return response.data[0].embedding

def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    embeddings = model.encode(texts, show_progress_bar=True)

    # embeddings is a 2D numpy array — convert each row to a plain list
    return [embedding.tolist() for embedding in embeddings]

    # texts = [text.replace("\n", " ") for text in texts]
    #
    # response = client.embeddings.create(
    #     input = texts,
    #     model = EMBEDDING_MODEL
    # )
    #
    # return [item.embedding for item in response.data]
