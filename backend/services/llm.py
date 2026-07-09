import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GOQ_API_KEY"))

MODEL = "llama-3.1-8b-instant"

def get_answer(question:str, context_chunks:list[dict]) -> dict:

    context = " "
    for i, chunk in enumerate(context_chunks):
        context += f"[Chunk {i + 1}]:\n{chunk['text']}\n\n"

    system_prompt = """You are a helpful assistant that answers questions based strictly on the provided document context.
    Rules:
    - Only use information from the context chunks provided
    - If the answer is not in the context, say "I could not find this information in the document"
    - Be concise and direct
    - Reference which chunk your answer came from when possible"""

    user_prompt = f"""Context from the document:
    {context}

    Question: {question}

    Answer based only on the context above:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,

        max_tokens=1024  # max length of the answer
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "model_used": MODEL,
        "chunks_used": len(context_chunks)
    }
