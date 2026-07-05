import os
from fastapi import APIRouter, UploadFile, File, File, HTTPException
from services.chunking import extract_text_from_pdf, split_into_chunks
from services.vector_store import store_chunks
from services.embedding import get_embeddings_batch

router = APIRouter()

Upload_DIR = "uploaded_files"
os.makedirs(Upload_DIR, exist_ok=True)

@router.post("/upload")
async def upload_document(file: UploadFile = File()):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_path = os.path.join(Upload_DIR, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    return{
        "filename": file.filename,
        "message": "File uploaded successfully",
        "saved_path": file_path
    }

@router.get("/chunk/{filename}")
def chunk_document(filename: str):

    file_path = os.path.join(Upload_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found. Upload it first!")

    extraction_result = extract_text_from_pdf(file_path)

    if not extraction_result["text"].strip():
        raise HTTPException(status_code=422, detail="Could not extract text from this")

    chunks = split_into_chunks(extraction_result["text"])


    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = get_embeddings_batch(chunk_texts)

    stored_count = store_chunks(filename, chunks, embeddings)

    return {
        "filename": filename,
        "extraction_method": extraction_result["method"],
        "chunks_stored": stored_count,
        "message": f"Document processed and ready to query"
    }