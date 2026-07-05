import pdfplumber
import tiktoken

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

def extract_text_from_pdf(file_path: str) -> dict:

    full_text = ""

    with pdfplumber.open(file_path) as pdf:

        for page in pdf.pages:

            text = page.extract_text()
            if text:
                full_text += text + "\n"

    return {
        "text": full_text,
        "method": "pdfplumber"
    }

def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:

    tokens = tokenizer.encode(text)

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(tokens):

        end = min(start + chunk_size, len(tokens))

        chunk_tokens = tokens[start:end]

        chunk_text = tokenizer.decode(chunk_tokens)

        chunks.append({
            "chunk_index": chunk_index,
            "text": chunk_text,
            "token_count": len(chunk_tokens)

        })

        start += chunk_size - overlap
        chunk_index += 1

    return chunks