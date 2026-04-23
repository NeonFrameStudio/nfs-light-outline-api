from io import BytesIO
from typing import List, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()


@app.get("/")
def home():
    return {
        "status": "working",
        "message": "NFS Light Outline API is live"
    }


@app.get("/health")
def health():
    return {"ok": True}


def decode_image(content: bytes) -> np.ndarray | None:
    arr = np.frombuffer(content, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def resize_image(img: np.ndarray, max_width: int = 1400) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    if w <= max_width:
        return img.copy(), 1.0

    scale = max_width / float(w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def clean_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def build_red_mask(hsv: np.ndarray) -> np.ndarray:
    lower_red1 = np.array([0, 70, 70], dtype=np.uint8)
    upper_red1 = np.array([12, 255, 255], dtype=np.uint8)

    lower_red2 = np.array([168, 70, 70], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    return cv2.bitwise_or(mask1, mask2)


def build_white_mask(hsv: np.ndarray, gray: np.ndarray) -> np.ndarray:
    lower_white = np.array([0, 0, 150], dtype=np.uint8)
    upper_white = np.array([180, 90, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_white, upper_white)

    _, mask_gray = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)

    return cv2.bitwise_and(mask_hsv, mask_gray)


def build_warm_mask(hsv: np.ndarray, gray: np.ndarray) -> np.ndarray:
    lower_warm = np.array([10, 40, 120], dtype=np.uint8)
    upper_warm = np.array([40, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_warm, upper_warm)

    _, mask_gray = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

    return cv2.bitwise_and(mask_hsv, mask_gray)


def get_contours(mask: np.ndarray) -> List[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def filter_rear_contours(contours: List[np.ndarray], w: int, h: int) -> List[np.ndarray]:
    kept: List[np.ndarray] = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < 120:
            continue
        if area > (w * h) * 0.18:
            continue

        x, y, cw, ch = cv2.boundingRect(c)
        cx = x + (cw / 2)
        cy = y + (ch / 2)

        if cw < 12 or ch < 12:
            continue

        # keep only outer left / outer right
        if not (cx < w * 0.35 or cx > w * 0.65):
            continue

        # rear lights usually in upper half to upper-middle
        if cy > h * 0.78:
            continue

        kept.append(c)

    return kept


def filter_front_contours(contours: List[np.ndarray], w: int, h: int) -> List[np.ndarray]:
    kept: List[np.ndarray] = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < 100:
            continue
        if area > (w * h) * 0.18:
            continue

        x, y, cw, ch = cv2.boundingRect(c)
        cx = x + (cw / 2)
        cy = y + (ch / 2)

        if cw < 12 or ch < 10:
            continue

        # keep only outer left / outer right
        if not (cx < w * 0.38 or cx > w * 0.62):
            continue

        # headlights usually sit around middle-ish vertical area, not roof/ground
        if cy < h * 0.18 or cy > h * 0.82:
            continue

        kept.append(c)

    return kept


def simplify_contour(c: np.ndarray) -> np.ndarray:
    epsilon = 0.006 * cv2.arcLength(c, True)
    return cv2.approxPolyDP(c, epsilon, True)


def draw_glow_outline(
    base: np.ndarray,
    contours: List[np.ndarray],
    color_bgr: Tuple[int, int, int],
    outline_thickness: int = 3,
    glow_sigma: int = 12,
) -> np.ndarray:
    if not contours:
        return base.copy()

    sharp = np.zeros_like(base)
    soft = np.zeros_like(base)

    simplified = [simplify_contour(c) for c in contours]

    cv2.drawContours(soft, simplified, -1, color_bgr, 10)
    cv2.drawContours(sharp, simplified, -1, color_bgr, outline_thickness)

    soft_blur = cv2.GaussianBlur(soft, (0, 0), glow_sigma)

    result = base.copy()
    result = cv2.addWeighted(result, 1.0, soft_blur, 0.85, 0)
    result = cv2.addWeighted(result, 1.0, sharp, 1.0, 0)

    return result


def process_rear(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red_mask = build_red_mask(hsv)
    red_mask = clean_mask(red_mask, kernel_size=5)

    contours = get_contours(red_mask)
    contours = filter_rear_contours(contours, img.shape[1], img.shape[0])

    return draw_glow_outline(
        img,
        contours,
        color_bgr=(0, 0, 255),
        outline_thickness=3,
        glow_sigma=10,
    )


def process_front(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    white_mask = build_white_mask(hsv, gray)
    warm_mask = build_warm_mask(hsv, gray)

    light_mask = cv2.bitwise_or(white_mask, warm_mask)
    light_mask = clean_mask(light_mask, kernel_size=5)

    contours = get_contours(light_mask)
    contours = filter_front_contours(contours, img.shape[1], img.shape[0])

    return draw_glow_outline(
        img,
        contours,
        color_bgr=(255, 255, 255),
        outline_thickness=3,
        glow_sigma=10,
    )


def auto_mode(img: np.ndarray) -> str:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    red_mask = build_red_mask(hsv)
    white_mask = build_white_mask(hsv, gray)
    warm_mask = build_warm_mask(hsv, gray)
    front_mask = cv2.bitwise_or(white_mask, warm_mask)

    red_pixels = int(cv2.countNonZero(red_mask))
    front_pixels = int(cv2.countNonZero(front_mask))

    if red_pixels > front_pixels * 1.15:
        return "rear"
    return "front"


@app.post("/preview")
async def preview(
    file: UploadFile = File(...),
    view: str = Form("auto"),
):
    content = await file.read()
    img = decode_image(content)

    if img is None:
        return JSONResponse({"error": "Could not read image"}, status_code=400)

    img, _ = resize_image(img, max_width=1400)

    view = (view or "auto").strip().lower()
    if view not in {"auto", "rear", "front"}:
        return JSONResponse(
            {"error": "view must be one of: auto, rear, front"},
            status_code=400,
        )

    if view == "auto":
        chosen_view = auto_mode(img)
    else:
        chosen_view = view

    if chosen_view == "rear":
        result = process_rear(img)
    else:
        result = process_front(img)

    ok, encoded = cv2.imencode(".png", result)
    if not ok:
        return JSONResponse({"error": "Failed to encode image"}, status_code=500)

    return StreamingResponse(
        BytesIO(encoded.tobytes()),
        media_type="image/png",
        headers={"X-NFS-View": chosen_view},
    )
