from io import BytesIO

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse

app = FastAPI()


@app.get("/")
def home():
    return {
        "status": "working",
        "message": "NFS Light Outline API is live"
    }


@app.post("/preview")
async def preview(file: UploadFile = File(...)):
    content = await file.read()
    return StreamingResponse(
        BytesIO(content),
        media_type=file.content_type or "application/octet-stream"
    )
