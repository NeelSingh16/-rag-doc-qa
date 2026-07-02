import os
from fastapi import APIRouter, UploadFile, File, File, HTTPException

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