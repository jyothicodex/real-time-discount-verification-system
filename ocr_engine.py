# utils/ocr_engine.py
# ─────────────────────────────────────────────
# Bill OCR Module for DeciBuy
# Extracts product names and prices from uploaded bill images
# Primary: EasyOCR | Fallback: pytesseract
# ─────────────────────────────────────────────

import re
from typing import Optional


# ── Try importing OCR libraries (optional dependencies) ──
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from PIL import Image
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False


def extract_text_from_image(image_bytes: bytes) -> Optional[str]:
    """
    Extract raw text from an image using available OCR engine.
    Tries EasyOCR first, then pytesseract, then returns None.

    Args:
        image_bytes: Image file as bytes (from st.file_uploader)

    Returns:
        Extracted text as string, or None if no OCR library available.
    """
    if EASYOCR_AVAILABLE:
        return _ocr_via_easyocr(image_bytes)
    elif PYTESSERACT_AVAILABLE:
        return _ocr_via_tesseract(image_bytes)
    else:
        return None


def _ocr_via_easyocr(image_bytes: bytes) -> str:
    """Use EasyOCR to extract text. Supports English + common Indian scripts."""
    import numpy as np
    import io
    from PIL import Image as PILImage

    image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(image)

    # gpu=False ensures it works on systems without CUDA
    reader = easyocr.Reader(["en"], gpu=False)
    results = reader.readtext(img_array, detail=0, paragraph=True)
    return "\n".join(results)


def _ocr_via_tesseract(image_bytes: bytes) -> str:
    """Use pytesseract as a fallback OCR engine."""
    import io
    from PIL import Image as PILImage

    image = PILImage.open(io.BytesIO(image_bytes))
    text = pytesseract.image_to_string(image)
    return text


def parse_bill_items(raw_text: str) -> list:
    """
    Parse OCR output into structured bill items.
    Attempts to extract (product_name, quantity, price) tuples.

    Strategy:
    - Look for lines with a price pattern (digits with ₹ or decimal)
    - Separate item name from price using regex
    - Return list of dicts

    Args:
        raw_text: Raw string from OCR

    Returns:
        List of dicts: [{name, quantity_g_ml, price, unit_price_per_100}]
    """
    items = []
    lines = raw_text.split("\n")

    # Regex: match a price at the end of a line (e.g., 120.00 or ₹45)
    price_pattern = re.compile(r"[₹Rs.]?\s*(\d+\.?\d*)\s*$")
    # Quantity pattern: e.g., 500g, 1kg, 200ml, 1L
    qty_pattern   = re.compile(r"(\d+\.?\d*)\s*(g|ml|kg|L|gm|Kg|KG|ltr|Ltr)\b", re.IGNORECASE)

    for line in lines:
        line = line.strip()
        if not line or len(line) < 3:
            continue

        price_match = price_pattern.search(line)
        if not price_match:
            continue

        price = float(price_match.group(1))
        if price <= 0 or price > 100000:   # sanity filter
            continue

        # Remove price from line to get product name
        name_part = line[:price_match.start()].strip()
        name_part = re.sub(r"[₹Rs.]+", "", name_part).strip()

        if len(name_part) < 2:
            continue

        # Try to extract quantity from the name part
        qty_match = qty_pattern.search(name_part)
        quantity_g_ml = None
        if qty_match:
            qty_val  = float(qty_match.group(1))
            qty_unit = qty_match.group(2).lower()
            # Normalize
            if qty_unit in ["kg", "ltr"]:
                quantity_g_ml = qty_val * 1000
            elif qty_unit == "l":
                quantity_g_ml = qty_val * 1000
            else:
                quantity_g_ml = qty_val

        # Calculate unit price only if quantity found
        unit_price = None
        if quantity_g_ml and quantity_g_ml > 0:
            unit_price = (price / quantity_g_ml) * 100

        items.append({
            "name":              name_part,
            "quantity_g_ml":     quantity_g_ml,
            "price":             price,
            "unit_price_per_100": round(unit_price, 2) if unit_price else None
        })

    return items


def ocr_status() -> dict:
    """
    Return the availability status of OCR engines for display in UI.
    """
    return {
        "easyocr":     EASYOCR_AVAILABLE,
        "pytesseract": PYTESSERACT_AVAILABLE,
        "any_ocr":     EASYOCR_AVAILABLE or PYTESSERACT_AVAILABLE
    }
