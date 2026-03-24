"""
DocDecode — Generic AI Vision Document Extraction Engine

Extracts structured data from ANY document type using Claude Vision.
Add new document types by simply adding a prompt to DOC_TYPES.

Supports: licence discs, driver's licences, vehicle registration,
ID documents, and any custom document type.

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
    """Resize large images and auto-rotate from EXIF."""
    media_type_map = {
        "image/jpeg": "image/jpeg", "image/jpg": "image/jpeg",
        "image/png": "image/png", "image/webp": "image/webp",
        "image/gif": "image/gif",
        "image/bmp": "image/png", "image/tiff": "image/png",
    }
    media_type = media_type_map.get(content_type, "image/jpeg")

    try:
        img = Image.open(BytesIO(image_bytes))
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
        max_dim = 2048
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        buf = BytesIO()
        fmt = "PNG" if media_type == "image/png" else "JPEG"
        img.save(buf, format=fmt, quality=90)
        image_bytes = buf.getvalue()
    except Exception:
        pass

    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    return b64, media_type


# ---------------------------------------------------------------------------
# Claude Vision call (shared by all document types)
# ---------------------------------------------------------------------------

def _call_vision(image_bytes: bytes, content_type: str, prompt: str) -> dict:
    """Send image + prompt to Claude Vision and return parsed JSON."""
    client = anthropic.Anthropic()
    b64_data, media_type = _prepare_image(image_bytes, content_type)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
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
                {"type": "text", "text": prompt},
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
        return json.loads(raw_text)
    except json.JSONDecodeError:
        logger.error("Failed to parse Claude response:\n%s", raw_text)
        return {"_raw_response": raw_text, "_parse_error": True}


# ═══════════════════════════════════════════════════════════════════════════
# DOCUMENT TYPE REGISTRY
# Each entry: { "name", "description", "prompt" }
# Add new documents here — no code changes needed elsewhere.
# ═══════════════════════════════════════════════════════════════════════════

DOC_TYPES = {}


def register_doc_type(key: str, name: str, description: str, prompt: str):
    """Register a new document type for extraction."""
    DOC_TYPES[key] = {
        "name": name,
        "description": description,
        "prompt": prompt,
    }


# ---------------------------------------------------------------------------
# 1. SA Licence Disc
# ---------------------------------------------------------------------------
register_doc_type(
    key="licence_disc",
    name="SA Licence Disc",
    description="South African vehicle licence disc (lisensieskyfie)",
    prompt="""You are an expert OCR system specialised in South African vehicle licence discs (lisensieskyfies).

CRITICAL — ROTATION HANDLING:
The disc image may be rotated 90°, 180°, or 270°, or at any angle. Before reading ANY text:
1. Look for the words "LISENSIESKYF" (usually curved along the top) and "LICENCE DISC" (usually at the bottom).
2. Use these anchor words to determine the correct orientation.
3. Mentally rotate the image so "LISENSIESKYF" is at the top and "LICENCE DISC" is at the bottom.
4. ONLY THEN read the fields in their correct orientation.

Known SA licence disc layout (top to bottom after correct rotation):
- Top curve: LISENSIESKYF
- RSA + NO. (disc number) + NR.
- Licence no. / Lisensienr: ........  Veh. register no. / Vrt.registernr: ........
- VIN: ........ (17 characters, e.g. WUAZZFX2H7904038)
- Engine no. / Enjinnr: ........
- Make / Fabrikaat: ........
- Description / Beskrywing: ........ (e.g. COUPE (open top))
- [barcode area]
- Date of test / Datum van toets: YYYY-MM-DD
- Persons / Persone: NNN   Seated / Sittende: NNN   Standing / Staande: NNN
- Date of expiry / Vervaldatum: YYYY-MM-DD
- Bottom: LICENCE DISC
- Right side labels: Fees/Gelde, GVM/BVM (in kg), Tare/Tarra (in kg), Fabrikaat, Beskrywing

Extract **every** field visible on the disc.

Return ONLY a JSON object (no markdown, no commentary) with exactly these keys.
If a field is not visible or unreadable, set its value to `null`.

{
  "disc_number": "The disc number after NO. (e.g. 1001056N64N8)",
  "licence_number": "Licence / Lisensie number (e.g. CAA272324 or 1MAGIN8WP)",
  "vehicle_register_number": "Vehicle registration / Vrt.registernr (e.g. WMH861W or HBH682K)",
  "vin": "Vehicle Identification Number — always exactly 17 characters (e.g. WUAZZFX2H7904038)",
  "engine_number": "Engine number / Enjinnr (e.g. CSP00683)",
  "make": "Vehicle make / Fabrikaat (e.g. MERCEDES-BENZ, AUDI, TOYOTA)",
  "description": "Vehicle description / Beskrywing (e.g. COUPE (closed top), COUPE (open top))",
  "fees": "Fees / Gelde amount — this is a small decimal number like 2.2, NOT a weight",
  "gvm_kg": "Gross Vehicle Mass — labelled GVM/BVM on the right side, in kg (e.g. 2020)",
  "tare_kg": "Tare weight — labelled Tare/Tarra on the right side, in kg (e.g. 1685)",
  "date_of_test": "Date of test / Datum van toets (YYYY-MM-DD)",
  "persons_seated": "Number of seated persons from Persone line (e.g. 2)",
  "persons_standing": "Number of standing persons from Sittende/Standing line (e.g. 0)",
  "date_of_expiry": "Date of expiry / Vervaldatum — the large date at bottom (YYYY-MM-DD)",
  "country": "Country code shown on disc (RSA)"
}

Important rules:
- ROTATION: You MUST correctly orient the image before reading. Misreading due to rotation is the #1 failure mode.
- Dates must be in YYYY-MM-DD format.
- fees is a small decimal (e.g. 2.2), NOT a year or weight.
- gvm_kg and tare_kg are 4-digit numbers (e.g. 2020, 1685).
- persons_seated comes from "Persone: NNN" (usually 002 = 2 people). persons_standing from "Sittende NNN" (usually 000 = 0).
- The VIN is always exactly 17 alphanumeric characters.
- Return ONLY the raw JSON object. No markdown fences, no explanation."""
)


# ---------------------------------------------------------------------------
# 2. SA Driver's Licence
# ---------------------------------------------------------------------------
register_doc_type(
    key="drivers_licence",
    name="SA Driver's Licence",
    description="South African driving licence card",
    prompt="""You are an expert OCR system specialised in South African driving licence cards.

The card may be photographed at any angle, rotated, or have glare. Handle this gracefully.

South African driving licence card layout:
- Top: "DRIVING LICENCE" and "SOUTH AFRICA" with SADC ZA logo
- "CARTA DE CONDUCAO" (Portuguese translation)
- Full name (surname and initials)
- ID Number (13-digit SA ID number, may have prefix like 02/)
- Gender: MALE or FEMALE
- Birth date: DD/MM/YYYY format followed by country code (ZA)
- Restriction code (usually 1 or 0)
- Licence Number: alphanumeric (e.g. 603900029FXH)
- No.: card number (usually 1, 2, etc.)
- Valid: start date - end date (DD/MM/YYYY - DD/MM/YYYY)
- Issued: country code (ZA)
- Code: licence code (A, A1, B, C, C1, EC, EC1, etc.)
- Vehicle restriction: number (usually 0)
- First Issue: DD/MM/YYYY
- Photo on the right side
- Signature at the bottom

Extract every field visible on the card.

Return ONLY a JSON object (no markdown, no commentary) with exactly these keys.
If a field is not visible or unreadable, set its value to null.

{
  "full_name": "Full name as printed (e.g. RH VENTER)",
  "id_number": "13-digit SA ID number only, without any prefix like 02/ (e.g. 9801135087080)",
  "id_number_full": "Full ID field as printed including prefix (e.g. 02/9801135087080)",
  "gender": "MALE or FEMALE",
  "date_of_birth": "Date of birth in YYYY-MM-DD format",
  "country": "Country code (e.g. ZA)",
  "restriction": "Restriction code number",
  "licence_number": "Licence number (e.g. 603900029FXH)",
  "card_number": "Card number from No. field (e.g. 1)",
  "valid_from": "Valid start date in YYYY-MM-DD format",
  "valid_to": "Valid end date in YYYY-MM-DD format",
  "issued_country": "Issued country code (e.g. ZA)",
  "licence_code": "Licence code (e.g. B, A, C1, EC)",
  "vehicle_restriction": "Vehicle restriction code (e.g. 0)",
  "first_issue_date": "First issue date in YYYY-MM-DD format"
}

Important rules:
- All dates must be converted to YYYY-MM-DD format (the card shows DD/MM/YYYY).
- id_number should be the raw 13-digit number WITHOUT any prefix.
- id_number_full should include the prefix exactly as printed.
- licence_code is the letter code (A, B, C, C1, EC, etc.), NOT the licence number.
- Numeric fields (restriction, card_number, vehicle_restriction) should be numbers, not strings.
- Return ONLY the raw JSON object. No markdown fences, no explanation."""
)


# ---------------------------------------------------------------------------
# 3. SA ID Document (Smart ID Card)
# ---------------------------------------------------------------------------
register_doc_type(
    key="id_document",
    name="SA ID Document",
    description="South African Smart ID card or green ID book",
    prompt="""You are an expert OCR system specialised in South African identity documents (Smart ID cards and green ID books).

The document may be photographed at any angle. Handle rotation and glare gracefully.

Extract every visible field from the ID document.

Return ONLY a JSON object (no markdown, no commentary) with these keys.
If a field is not visible or unreadable, set its value to null.

{
  "document_type": "SMART_ID or ID_BOOK",
  "surname": "Surname / Van",
  "names": "Full first names",
  "id_number": "13-digit SA ID number",
  "gender": "MALE or FEMALE",
  "date_of_birth": "YYYY-MM-DD format",
  "country_of_birth": "Country of birth (e.g. South Africa)",
  "citizenship": "Citizenship status (e.g. SA Citizen)",
  "nationality": "Nationality code (e.g. RSA)",
  "date_of_issue": "YYYY-MM-DD format",
  "smart_id_number": "Smart ID card number (if applicable)"
}

Rules:
- All dates in YYYY-MM-DD format.
- id_number is exactly 13 digits.
- Return ONLY the raw JSON. No markdown, no explanation."""
)


# ---------------------------------------------------------------------------
# 4. SA Vehicle Registration Certificate (NaTIS)
# ---------------------------------------------------------------------------
register_doc_type(
    key="vehicle_registration",
    name="SA Vehicle Registration Certificate",
    description="South African vehicle registration certificate (NaTIS document)",
    prompt="""You are an expert OCR system for South African vehicle registration certificates (NaTIS documents).

Extract every visible field from the registration certificate.

Return ONLY a JSON object (no markdown, no commentary) with these keys.
If a field is not visible or unreadable, set its value to null.

{
  "registration_number": "Vehicle registration number",
  "vin": "Vehicle Identification Number (17 characters)",
  "engine_number": "Engine number",
  "make": "Vehicle make",
  "model": "Vehicle model",
  "colour": "Vehicle colour",
  "year_of_manufacture": "Year of manufacture",
  "tare_kg": "Tare mass in kg",
  "gvm_kg": "Gross vehicle mass in kg",
  "owner_name": "Registered owner name",
  "owner_id_number": "Owner ID number",
  "owner_address": "Owner address",
  "registration_date": "Date of registration (YYYY-MM-DD)",
  "licence_expiry_date": "Licence expiry date (YYYY-MM-DD)",
  "transaction_number": "NaTIS transaction number"
}

Rules:
- All dates in YYYY-MM-DD format.
- Weights as numbers without 'kg' suffix.
- Return ONLY the raw JSON. No markdown, no explanation."""
)


# ---------------------------------------------------------------------------
# 5. Invoice / Receipt
# ---------------------------------------------------------------------------
register_doc_type(
    key="invoice",
    name="Invoice / Receipt",
    description="Invoice, receipt, or billing document",
    prompt="""You are an expert document extraction system for invoices and receipts.

Extract every visible field from the document.

Return ONLY a JSON object (no markdown, no commentary) with these keys.
If a field is not visible or unreadable, set its value to null.

{
  "vendor_name": "Company or vendor name",
  "vendor_address": "Vendor address",
  "vendor_vat_number": "VAT registration number",
  "vendor_phone": "Phone number",
  "vendor_email": "Email address",
  "invoice_number": "Invoice or receipt number",
  "invoice_date": "Invoice date (YYYY-MM-DD)",
  "due_date": "Payment due date (YYYY-MM-DD)",
  "customer_name": "Customer / bill-to name",
  "customer_address": "Customer address",
  "subtotal": "Subtotal amount before tax",
  "vat_amount": "VAT / tax amount",
  "total": "Total amount due",
  "currency": "Currency code (e.g. ZAR, USD)",
  "payment_method": "Payment method if shown",
  "line_items": [
    {
      "description": "Item description",
      "quantity": "Quantity",
      "unit_price": "Price per unit",
      "amount": "Line total"
    }
  ]
}

Rules:
- All dates in YYYY-MM-DD format.
- Monetary values as numbers (no currency symbols).
- Return ONLY the raw JSON. No markdown, no explanation."""
)


# ---------------------------------------------------------------------------
# 6. Generic / Custom document (catch-all)
# ---------------------------------------------------------------------------
register_doc_type(
    key="generic",
    name="Generic Document",
    description="Any document — extracts all visible text and fields",
    prompt="""You are an expert document extraction system.

Analyse this document image carefully. Extract ALL visible text, fields, and data.

Return ONLY a JSON object (no markdown, no commentary) with these keys:

{
  "document_type": "Your best guess at the document type (e.g. invoice, contract, certificate, letter, form, ID card, etc.)",
  "title": "Document title or heading if visible",
  "date": "Any date on the document (YYYY-MM-DD format) or null",
  "fields": {
    "field_name": "value"
  }
}

Put ALL extracted key-value pairs inside the "fields" object, using the label/heading as the key and the extracted value as the value. Be thorough — extract every piece of readable text.

Rules:
- All dates in YYYY-MM-DD format.
- Numbers as numbers, not strings.
- Return ONLY the raw JSON. No markdown, no explanation."""
)


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def get_supported_doc_types() -> dict:
    """Return a dict of supported document types with descriptions."""
    return {
        key: {"name": dt["name"], "description": dt["description"]}
        for key, dt in DOC_TYPES.items()
    }


def extract_document(image_bytes: bytes, doc_type: str,
                     content_type: str = "image/jpeg") -> dict:
    """
    Extract data from any supported document type.

    Args:
        image_bytes: Raw image file bytes.
        doc_type: Document type key (e.g. 'licence_disc', 'drivers_licence', 'generic').
        content_type: MIME type of the image.

    Returns:
        Dict with extracted fields. Unreadable fields are None.
    """
    if doc_type not in DOC_TYPES:
        return {
            "_error": f"Unknown document type: '{doc_type}'",
            "_supported_types": list(DOC_TYPES.keys()),
        }

    prompt = DOC_TYPES[doc_type]["prompt"]
    return _call_vision(image_bytes, content_type, prompt)


# ═══════════════════════════════════════════════════════════════════════════
# Backward-compatible wrappers (existing endpoints still work)
# ═══════════════════════════════════════════════════════════════════════════

def extract_licence_disc(image_bytes: bytes, content_type: str = "image/jpeg") -> dict:
    return extract_document(image_bytes, "licence_disc", content_type)


def extract_drivers_licence(image_bytes: bytes, content_type: str = "image/jpeg") -> dict:
    return extract_document(image_bytes, "drivers_licence", content_type)


def extract_licence_disc_debug(image_bytes: bytes, content_type: str = "image/jpeg") -> dict:
    result = extract_document(image_bytes, "licence_disc", content_type)
    return {"result": result, "_debug": {}}
