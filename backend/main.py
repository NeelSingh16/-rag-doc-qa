from fastapi import FastAPI
from routers import upload

app = FastAPI(title = "RAG Document Q&A")
app.include_router(upload.router)

@app.get("/")
def health_check():
    return{"status": "ok"}

