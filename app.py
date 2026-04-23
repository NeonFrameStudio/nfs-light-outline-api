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
        "message": "NFS Light Outline API V1 is live"
    }


@app.get("/health")
def health():
    return {"ok": True}


def decode_image(content: bytes):
    arr = np.frombuffer(content, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def resize_image(img: np.ndarray, max_width: int = 1400) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_width:
        return img.copy()

    scale = max_width / float(w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def clean_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def build_red_mask(hsv: np.ndarray) -> np.ndarray:
    lower_red1 = np.array([0, 55, 45], dtype=np.uint8)
    upper_red1 = np.array([15, 255, 255], dtype=np.uint8)

    lower_red2 = np.array([165, 55, 45], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    return cv2.bitwise_or(mask1, mask2)


def build_front_mask(hsv: np.ndarray, gray: np.ndarray) -> np.ndarray:
    lower_white = np.array([0, 0, 135], dtype=np.uint8)
    upper_white = np.array([180, 110, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    lower_warm = np.array([8, 20, 120], dtype=np.uint8)
    upper_warm = np.array([40, 255, 255], dtype=np.uint8)
    warm_mask = cv2.inRange(hsv, lower_warm, upper_warm)

    _, bright_mask = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)

    combined = cv2.bitwise_or(white_mask, warm_mask)
    combined = cv2.bitwise_and(combined, bright_mask)
    return combined


def get_contours(mask: np.ndarray) -> List[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def simplify_contour(c: np.ndarray) -> np.ndarray:
    epsilon = 0.004 * cv2.arcLength(c, True)
    return cv2.approxPolyDP(c, epsilon, True)


def sort_by_area(contours: List[np.ndarray]) -> List[np.ndarray]:
    return sorted(contours, key=cv2.contourArea, reverse=True)


def contour_center(c: np.ndarray) -> Tuple[float, float]:
    x, y, w, h = cv2.boundingRect(c)
    return x + (w / 2), y + (h / 2)


def filter_rear_contours(contours: List[np.ndarray], w: int, h: int) -> List[np.ndarray]:
    kept = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < 80:
            continue
        if area > (w * h) * 0.30:
            continue

        x, y, cw, ch = cv2.boundingRect(c)
        cx = x + (cw / 2)
        cy = y + (ch / 2)

        if cw < 8 or ch < 8:
            continue

        # rear lights usually not dead center
        if 0.42 * w < cx < 0.58 * w:
            continue

        # ignore very bottom junk
        if cy > h * 0.90:
            continue

        kept.append(c)

    kept = sort_by_area(kept)

    # keep the biggest few so it actually does something today
    return kept[:6]


def filter_front_contours(contours: List[np.ndarray], w: int, h: int) -> List[np.ndarray]:
    kept = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < 70:
            continue
        if area > (w * h) * 0.25:
            continue

        x, y, cw, ch = cv2.boundingRect(c)
        cx = x + (cw / 2)
        cy = y + (ch / 2)

        if cw < 8 or ch < 6:
            continue

        # ignore exact center grill area
        if 0.44 * w < cx < 0.56 * w:
            continue

        # ignore roof/ground junk
        if cy < h * 0.12 or cy > h * 0.88:
            continue

        kept.append(c)

    kept = sort_by_area(kept)
    return kept[:6]


def fallback_outer_two(contours: List[np.ndarray], w: int) -> List[np.ndarray]:
    if not contours:
        return []

    left = None
    right = None

    for c in contours:
        cx, _ = contour_center(c)
        if cx < w / 2:
            if left is None or cv2.contourArea(c) > cv2.contourArea(left):
                left = c
        else:
            if right is None or cv2.contourArea(c) > cv2.contourArea(right):
                right = c

    result = []
    if left is not None:
        result.append(left)
    if right is not None:
        result.append(right)
    return result


def draw_glow_outline(
    base: np.ndarray,
    contours: List[np.ndarray],
    color_bgr: Tuple[int, int, int],
    outline_thickness: int = 4,
    glow_thickness: int = 12,
    glow_sigma: int = 14,
) -> np.ndarray:
    if not contours:
        return base.copy()

    sharp = np.zeros_like(base)
    soft = np.zeros_like(base)

    simplified = [simplify_contour(c) for c in contours]

    cv2.drawContours(soft, simplified, -1, color_bgr, glow_thickness)
    cv2.drawContours(sharp, simplified, -1, color_bgr, outline_thickness)

    soft_blur = cv2.GaussianBlur(soft, (0, 0), glow_sigma)

    result = base.copy()
    result = cv2.addWeighted(result, 1.0, soft_blur, 1.0, 0)
    result = cv2.addWeighted(result, 1.0, sharp, 1.0, 0)
    return result


def process_rear(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    red_mask = build_red_mask(hsv)
    red_mask = clean_mask(red_mask, kernel_size=5)

    contours = get_contours(red_mask)
    filtered = filter_rear_contours(contours, img.shape[1], img.shape[0])

    if not filtered:
        filtered = fallback_outer_two(sort_by_area(contours), img.shape[1])

    return draw_glow_outline(
        img,
        filtered,
        color_bgr=(0, 0, 255),
        outline_thickness=4,
        glow_thickness=14,
        glow_sigma=16,
    )


def process_front(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    front_mask = build_front_mask(hsv, gray)
    front_mask = clean_mask(front_mask, kernel_size=5)

    contours = get_contours(front_mask)
    filtered = filter_front_contours(contours, img.shape[1], img.shape[0])

    if not filtered:
        filtered = fallback_outer_two(sort_by_area(contours), img.shape[1])

    return draw_glow_outline(
        img,
        filtered,
        color_bgr=(255, 255, 255),
        outline_thickness=4,
        glow_thickness=14,
        glow_sigma=16,
    )


def auto_mode(img: np.ndarray) -> str:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    red_mask = build_red_mask(hsv)
    front_mask = build_front_mask(hsv, gray)

    red_pixels = int(cv2.countNonZero(red_mask))
    front_pixels = int(cv2.countNonZero(front_mask))

    if red_pixels > front_pixels * 0.9:
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

    img = resize_image(img, max_width=1400)

    view = (view or "auto").strip().lower()
    if view not in {"auto", "rear", "front"}:
        return JSONResponse(
            {"error": "view must be one of: auto, rear, front"},
            status_code=400,
        )

    chosen_view = auto_mode(img) if view == "auto" else view

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
