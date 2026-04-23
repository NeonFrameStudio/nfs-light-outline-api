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

    arr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Could not read image"}

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # red taillight mask
    lower_red1 = np.array([0, 70, 70])
    upper_red1 = np.array([12, 255, 255])

    lower_red2 = np.array([168, 70, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # clean mask
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    # find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # blank overlay for outline
    overlay = np.zeros_like(img)

    h, w = img.shape[:2]

    for c in contours:
        area = cv2.contourArea(c)

        if area < 150:
            continue

        x, y, cw, ch = cv2.boundingRect(c)

        if area > (w * h) * 0.2:
            continue

        cx = x + (cw / 2)

        # only keep contours on outer left / outer right
        if not (cx < w * 0.35 or cx > w * 0.65):
            continue

        # ignore tiny flat junk
        if cw < 12 or ch < 12:
            continue

        cv2.drawContours(overlay, [c], -1, (0, 0, 255), 3)

    # soft glow from the outline only
    glow = cv2.GaussianBlur(overlay, (0, 0), 8)

    # merge glow + sharp outline back onto original
    result = img.copy()
    result = cv2.addWeighted(result, 1.0, glow, 0.9, 0)
    result = cv2.addWeighted(result, 1.0, overlay, 1.0, 0)

    _, buffer = cv2.imencode(".png", result)

    return StreamingResponse(
        BytesIO(buffer.tobytes()),
        media_type="image/png"
    )
