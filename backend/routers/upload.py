import os
from fastapi import APIRouter, UploadFile, File, File, HTTPException
from services.chunking import extract_text_from_pdf, split_into_chunks

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

    raw_text = extract_text_from_pdf(file_path)

    chunks = split_into_chunks(raw_text)

    return {
        "filename": filename,
        "total_chunks": len(chunks),
        "chunks": chunks
    }