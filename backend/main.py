from fastapi import FastAPI
from routers import upload,query

app = FastAPI(title = "RAG Document Q&A")
app.include_router(upload.router)
app.include_router(query.router)

@app.get("/")
def health_check():
    return{"status": "ok"}

