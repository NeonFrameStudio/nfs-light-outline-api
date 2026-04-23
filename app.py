from io import BytesIO

import cv2
import numpy as np
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

    # convert upload → OpenCV image
    arr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Could not read image"}

    # convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # detect red taillight areas (2 red ranges)
    lower_red1 = np.array([0, 80, 80])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 80, 80])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = cv2.bitwise_or(mask1, mask2)

    # create red glow effect
    glow = cv2.GaussianBlur(red_mask, (0, 0), 15)

    # overlay red glow onto original
    result = img.copy()
    result[:, :, 2] = cv2.addWeighted(
        result[:, :, 2],
        1.0,
        glow,
        0.8,
        0
    )

    # return final image
    _, buffer = cv2.imencode(".png", result)

    return StreamingResponse(
        BytesIO(buffer.tobytes()),
        media_type="image/png"
    )
