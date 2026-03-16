"""
Claude Vision extraction engine for South African licence discs.

Sends the disc image to Claude's vision API which handles:
- Any rotation or angle
- Glare and poor lighting
- Different disc layouts and provinces
- Structured JSON output in one shot

Requires: ANTHROPIC_API_KEY environment variable
"""

import os
import json
import base64
import logging
from io import BytesIO

import anthropic
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def _prepare_image(image_bytes: bytes, content_type: str) -> tuple[str, str]:
    """Resize large images and return (base64_data, media_type)."""
    media_type_map = {
        "image/jpeg": "image/jpeg", "image/jpg": "image/jpeg",
        "image/png": "image/png", "image/webp": "image/webp",
        "image/gif": "image/gif",
        "image/bmp": "image/png", "image/tiff": "image/png",
    }
    media_type = media_type_map.get(content_type, "image/jpeg")

    try:
        img = Image.open(BytesIO(image_bytes))
        # Auto-rotate based on EXIF data (phone photos)
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
        # Resize if too large (keeps API costs down)
        max_dim = 2048
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        buf = BytesIO()
        fmt = "PNG" if media_type == "image/png" else "JPEG"
        img.save(buf, format=fmt, quality=90)
        image_bytes = buf.getvalue()
    except Exception:
        pass  # If PIL fails, send raw bytes

    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    return b64, media_type


# ---------------------------------------------------------------------------
# Claude Vision prompt
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are an expert OCR system specialised in South African vehicle licence discs (lisensieskyfies).

The image may be rotated, angled, or have glare — handle this gracefully.

Analyse the image carefully and extract **every** field visible on the disc.

Return ONLY a JSON object (no markdown, no commentary) with exactly these keys.
If a field is not visible or unreadable, set its value to `null`.

{
  "disc_number": "The licence disc number (after NO.)",
  "licence_number": "Licence / Lisensie number (e.g. CAA272324 or 1MAGIN8WP)",
  "vehicle_register_number": "Vehicle registration / Vrt.registernr (e.g. WMH861W or HBH682K)",
  "vin": "Vehicle Identification Number (17-char VIN)",
  "engine_number": "Engine number / Enjinnr (e.g. CSP00683)",
  "make": "Vehicle make / Fabrikaat (e.g. MERCEDES-BENZ, AUDI, TOYOTA)",
  "description": "Vehicle description / Beskrywing (e.g. Coupe (closed top), Coupe (open top))",
  "fees": "Fees / Gelde amount",
  "gvm_kg": "Gross Vehicle Mass in kg",
  "tare_kg": "Tare mass in kg",
  "date_of_test": "Date of test / Datum van toets (YYYY-MM-DD)",
  "persons_seated": "Number of seated persons",
  "persons_standing": "Number of standing persons",
  "date_of_expiry": "Date of expiry / Vervaldatum (YYYY-MM-DD)",
  "country": "Country code shown on disc (e.g. RSA)"
}

Important:
- Dates must be in YYYY-MM-DD format.
- Numeric fields (fees, gvm_kg, tare_kg, persons_seated, persons_standing) should be numbers, not strings.
- If a number has a unit suffix like "kg" on the disc, strip it and return the bare number.
- Return ONLY the raw JSON object. No markdown fences, no explanation."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_licence_disc(image_bytes: bytes, content_type: str = "image/jpeg") -> dict:
    """
    Extract all fields from a SA licence disc image using Claude Vision.

    Args:
        image_bytes: Raw image file bytes.
        content_type: MIME type of the image.

    Returns:
        Dict with extracted fields. Unreadable fields are None.
    """
    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
    b64_data, media_type = _prepare_image(image_bytes, content_type)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64_data,
                    },
                },
                {"type": "text", "text": EXTRACTION_PROMPT},
            ],
        }],
    )

    raw_text = message.content[0].text.strip()

    # Strip possible markdown fences
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1]
    if raw_text.endswith("```"):
        raw_text = raw_text.rsplit("```", 1)[0]
    raw_text = raw_text.strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        logger.error("Failed to parse Claude response:\n%s", raw_text)
        return {"_raw_response": raw_text, "_parse_error": True}


def extract_licence_disc_debug(image_bytes: bytes, content_type: str = "image/jpeg") -> dict:
    """Same as extract_licence_disc but includes raw response for debugging."""
    client = anthropic.Anthropic()
    b64_data, media_type = _prepare_image(image_bytes, content_type)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64_data,
                    },
                },
                {"type": "text", "text": EXTRACTION_PROMPT},
            ],
        }],
    )

    raw_text = message.content[0].text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1]
    if raw_text.endswith("```"):
        raw_text = raw_text.rsplit("```", 1)[0]
    raw_text = raw_text.strip()

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        result = {"_raw_response": raw_text, "_parse_error": True}

    return {
        "result": result,
        "_debug": {"raw_response": raw_text},
    }
