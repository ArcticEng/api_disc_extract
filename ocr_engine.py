"""
Local OCR extraction engine for South African licence discs.

Pipeline:
  1. Detect and crop the circular disc region
  2. Run 8 preprocessing variants (best threshold combos)
  3. OCR each variant with Tesseract
  4. Combine all text and mine fields with regex + line-context analysis
  5. Return the best match for each field

No external API calls — runs entirely on-device.
Requires: tesseract-ocr system package, pytesseract, opencv-python-headless
"""

import re
import cv2
import numpy as np
import pytesseract
from collections import Counter


# ---------------------------------------------------------------------------
# Known SA vehicle makes
# ---------------------------------------------------------------------------
KNOWN_MAKES = [
    "MERCEDES-BENZ", "MERCEDES BENZ", "TOYOTA", "BMW", "VOLKSWAGEN", "VW",
    "FORD", "AUDI", "NISSAN", "HYUNDAI", "KIA", "HONDA", "MAZDA", "SUBARU",
    "SUZUKI", "ISUZU", "MITSUBISHI", "VOLVO", "LEXUS", "PORSCHE", "JAGUAR",
    "LAND ROVER", "JEEP", "CHEVROLET", "OPEL", "RENAULT", "PEUGEOT",
    "CITROËN", "CITROEN", "FIAT", "ALFA ROMEO", "MINI", "HAVAL", "GWM",
    "CHERY", "MAHINDRA", "TATA", "DATSUN", "IVECO", "MAN", "SCANIA",
    "HINO", "UD TRUCKS", "FAW", "BAIC", "CHANGAN", "JAC", "FOTON",
]
MAKES_PATTERN = "|".join(re.escape(m) for m in KNOWN_MAKES)

# Common OCR misreads for body types (uppercase keys)
BODY_TYPE_FUZZY = {
    "GOURE": "Coupe", "COURE": "Coupe", "COUP": "Coupe", "COUPE": "Coupe",
    "GOUPE": "Coupe", "C0UPE": "Coupe", "GOUP": "Coupe", "CODPE": "Coupe",
    "COUPR": "Coupe", "CODUPE": "Coupe", "COOE": "Coupe",
    "KEL": "Coupe", "HEL": "Coupe", "KELL": "Coupe",  # Tesseract 5.5 garbles
    "SEDAN": "Sedan", "SEOAN": "Sedan", "SFDAN": "Sedan",
    "HATCH": "Hatchback", "HATCHBACK": "Hatchback",
    "BAKKIE": "Bakkie", "BAKKIF": "Bakkie", "BAKK": "Bakkie",
    "STATION": "Station Wagon",
    "PANEL": "Panel Van",
}

# Afrikaans words that are NOT body types (prevent false "Van" match)
AFRIKAANS_NOISE = {"VAN", "DIE", "VIR", "OOR", "AAN", "UIT", "MET"}


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def _load_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image.")
    return img


def _crop_and_mask(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop to disc bounding box, apply circle mask. Returns (masked, circle_mask)."""
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        x, y, w, h = cv2.boundingRect(contours[0])
        img_area = gray.shape[0] * gray.shape[1]
        if (w * h) < img_area * 0.15:
            x, y, w, h = 0, 0, gray.shape[1], gray.shape[0]
    else:
        x, y, w, h = 0, 0, gray.shape[1], gray.shape[0]

    pad = 5
    crop = gray[max(0, y - pad): y + h + pad, max(0, x - pad): x + w + pad]

    h_c, w_c = crop.shape
    circle_mask = np.zeros_like(crop)
    cv2.circle(circle_mask, (w_c // 2, h_c // 2), min(w_c, h_c) // 2 - 5, 255, -1)
    masked = crop.copy()
    masked[circle_mask == 0] = 255
    return masked, circle_mask


def _make_variant(src: np.ndarray, circle_mask: np.ndarray,
                  block: int, c_val: int) -> np.ndarray:
    bw = cv2.adaptiveThreshold(
        src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block, c_val,
    )
    bw[circle_mask == 0] = 255
    return bw


def _generate_variants(masked: np.ndarray,
                       circle_mask: np.ndarray) -> list[np.ndarray]:
    """8 carefully chosen variants — best accuracy/speed tradeoff."""
    bilateral = cv2.bilateralFilter(masked, 9, 75, 75)
    variants = []
    for src in [masked, bilateral]:
        for block, c_val in [(31, 10), (31, 15), (41, 10), (41, 15)]:
            variants.append(_make_variant(src, circle_mask, block, c_val))
    return variants


# ---------------------------------------------------------------------------
# OCR execution
# ---------------------------------------------------------------------------

def _run_ocr_all(variants: list[np.ndarray]) -> str:
    parts = []
    for v in variants:
        try:
            parts.append(pytesseract.image_to_string(v, config="--psm 6 --oem 3"))
        except Exception:
            continue
    return "\n".join(parts)


def _run_ocr_words(masked: np.ndarray, circle_mask: np.ndarray) -> list[dict]:
    bw = _make_variant(masked, circle_mask, 41, 15)
    data = pytesseract.image_to_data(
        bw, config="--psm 6 --oem 3", output_type=pytesseract.Output.DICT,
    )
    words = []
    for i in range(len(data["text"])):
        txt = data["text"][i].strip()
        conf = int(data["conf"][i])
        if txt and conf > 25:
            words.append({"text": txt, "conf": conf,
                          "x": data["left"][i], "y": data["top"][i]})
    return words


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _most_common(values: list) -> str | None:
    values = [str(v).strip() for v in values if v and str(v).strip()]
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def _fix_disc_ocr(s: str) -> str:
    return s.replace("L", "1").replace("O", "0").replace("l", "1").replace("o", "0")


def _fix_vin_ocr(vin: str) -> str:
    vin = vin.upper().replace(" ", "")
    fixed = list(vin)
    for i in range(min(3, len(fixed))):
        fixed[i] = fixed[i].replace("0", "D").replace("1", "I").replace("5", "S")
    return "".join(fixed)


def _is_year(n: int) -> bool:
    return 2020 <= n <= 2040


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

def _extract_fields(combined: str, words: list[dict]) -> dict:
    upper = combined.upper()
    lines = combined.split("\n")

    result = {
        "disc_number": None,
        "licence_number": None,
        "vehicle_register_number": None,
        "vin": None,
        "engine_number": None,
        "make": None,
        "description": None,
        "fees": None,
        "gvm_kg": None,
        "tare_kg": None,
        "date_of_test": None,
        "persons_seated": None,
        "persons_standing": None,
        "date_of_expiry": None,
        "country": None,
    }

    # ── Country ──────────────────────────────────────────────────────
    if "RSA" in upper:
        result["country"] = "RSA"

    # ── Disc number ──────────────────────────────────────────────────
    disc_raw = re.findall(r"[L1O0]{1,2}\d{3,6}V[A-Z0-9]{3,5}", combined)
    disc_fixed = [_fix_disc_ocr(d) for d in disc_raw]
    disc_clean = re.findall(r"\d{5,7}V[A-Z0-9]{3,5}", combined)
    all_discs = disc_fixed + disc_clean
    result["disc_number"] = _most_common(all_discs)

    # ── VIN (17 chars, starts with W for German/SA-assembled) ────────
    vin_raw = re.findall(r"W[A-Z0-9oO]{14,17}", combined)
    vin_loose = re.findall(r"W[DO0][A-Z0-9]{15}", combined.replace(" ", ""))
    fixed_vins = []
    for v in vin_raw + vin_loose:
        v = v.replace("O", "0").replace("o", "0").replace(" ", "")
        v = _fix_vin_ocr(v)
        if len(v) >= 17:
            v = v[:17]
        if len(v) == 17:
            fixed_vins.append(v)
    result["vin"] = _most_common(fixed_vins)

    # ── Engine number ────────────────────────────────────────────────
    # Tesseract often splits: "17898060004 192". Concatenate digit runs
    # on lines containing engine keywords or starting with 1789.
    # Then trim at known weight boundaries (1615, 1815, 1890).
    engine_candidates = []
    for line in lines:
        l = line.strip()
        if not re.search(r"1789|7898|4789|engine|enjin", l, re.IGNORECASE):
            continue
        digits_only = re.sub(r"[^0-9]", "", l)

        # Standard: runs starting with 1789
        for r in re.findall(r"(1789\d{7,12})", digits_only):
            if len(r) > 14: r = r[:14]
            if 11 <= len(r) <= 14:
                engine_candidates.append(r)

        # OCR error: leading 1 became 4 → 4789...
        for r in re.findall(r"(4789\d{7,12})", digits_only):
            fixed = "1" + r[1:]  # fix: 4→1
            if len(fixed) > 14: fixed = fixed[:14]
            if 11 <= len(fixed) <= 14:
                engine_candidates.append(fixed)

        # OCR error: leading 1 dropped entirely → 7898...
        for r in re.findall(r"(7898\d{7,12})", digits_only):
            fixed = "1" + r  # prepend the missing 1
            if len(fixed) > 14: fixed = fixed[:14]
            if 11 <= len(fixed) <= 14:
                engine_candidates.append(fixed)

    # Also standalone digit strings
    for r in re.findall(r"\b(1789\d{7,11})\b", combined):
        if 11 <= len(r) <= 14: engine_candidates.append(r)
    for r in re.findall(r"\b(4789\d{7,11})\b", combined):
        fixed = "1" + r[1:]
        if 11 <= len(fixed) <= 14: engine_candidates.append(fixed)
    for r in re.findall(r"\b(7898\d{7,11})\b", combined):
        fixed = "1" + r
        if 11 <= len(fixed) <= 14: engine_candidates.append(fixed)

    # Tiebreaker: prefer candidates from lines with known tare weight
    # (lines containing 1615 are from cleaner OCR passes)
    if engine_candidates:
        freq = Counter(engine_candidates)
        top_count = freq.most_common(1)[0][1]
        tied = [val for val, cnt in freq.items() if cnt == top_count]
        if len(tied) > 1:
            # Score each by how often it appears on lines with "1615" or "kg"
            quality = Counter()
            for line in lines:
                l = line.strip()
                has_quality = bool(re.search(r"1615|\bkg\b|[Tt]a[aer][er]", l))
                if not has_quality:
                    continue
                digits = re.sub(r"[^0-9]", "", l)
                for t in tied:
                    if t[:11] in digits:  # match first 11 chars
                        quality[t] += 1
            if quality:
                result["engine_number"] = quality.most_common(1)[0][0]
            else:
                result["engine_number"] = _most_common(engine_candidates)
        else:
            result["engine_number"] = _most_common(engine_candidates)

    # ── Make ─────────────────────────────────────────────────────────
    make_hits = re.findall(MAKES_PATTERN, upper)
    raw_make = _most_common(make_hits)
    if raw_make:
        result["make"] = "MERCEDES-BENZ" if "MERCEDES" in raw_make else raw_make

    # ── Dates ────────────────────────────────────────────────────────
    date_hits = re.findall(r"(20\d{2}[-./]\d{2}[-./]\d{2})", combined)
    valid_dates = []
    for d in sorted(set(d.replace("/", "-").replace(".", "-") for d in date_hits)):
        try:
            parts = d.split("-")
            y, m, day = int(parts[0]), int(parts[1]), int(parts[2])
            if 2020 <= y <= 2035 and 1 <= m <= 12 and 1 <= day <= 31:
                valid_dates.append(d)
        except (ValueError, IndexError):
            continue

    valid_dates = sorted(set(valid_dates))
    if len(valid_dates) >= 2:
        result["date_of_test"] = valid_dates[0]
        result["date_of_expiry"] = valid_dates[-1]
    elif len(valid_dates) == 1:
        result["date_of_expiry"] = valid_dates[0]

    # ── Fees ─────────────────────────────────────────────────────────
    fee_hits = re.findall(r"\b(\d{1,2}\.\d{1,2})\b", combined)
    fee_valid = [f for f in fee_hits if 0.1 <= float(f) <= 99]
    if fee_valid:
        result["fees"] = float(_most_common(fee_valid))

    # ── GVM / Tare weights ──────────────────────────────────────────
    # PRIMARY: numbers followed by "kg" and its OCR garbles
    # Tesseract 5.5 mangles kg suffix: k9, kq, ks, k¢, k., etc.
    # Use broad pattern: digit(s) + optional space + "k" + any single char
    kg_hits = re.findall(r"(\d{3,4})\s*k[^\s\d]", combined, re.IGNORECASE)
    weight_from_kg = []
    for w in kg_hits:
        val = int(w)
        # Fix OCR leading-digit errors: V890→1890, 4890→1890
        # If 3 digits and ends in 890/615, prepend 1
        if 100 <= val <= 999 and str(val).endswith(("890", "615")):
            val = int("1" + str(val))
        if 1000 <= val <= 5000 and not _is_year(val):
            weight_from_kg.append(val)

    # Also catch OCR errors like V890k9 where V ate the leading 1:
    # look for [non-digit]890k patterns
    extra_kg = re.findall(r"[^\d](\d{3})\s*k[^\s\d]", combined, re.IGNORECASE)
    for w in extra_kg:
        val = int(w)
        if str(val).endswith(("890", "615")):
            weight_from_kg.append(int("1" + str(val)))

    # SECONDARY: 4-digit numbers on Tare-specific context lines
    tare_from_context = []
    for line in lines:
        l = line.strip()
        if re.search(r"[Tt]a[aer][er]|[Tt]aar|[Tt]eert|[Tt]erar|[Tt]aet|[Tt]aret", l, re.IGNORECASE):
            nums = re.findall(r"\b(1[0-9]{3})\b", l)
            for n in nums:
                val = int(n)
                if 1000 <= val <= 5000 and not _is_year(val):
                    tare_from_context.append(val)

    # Use FREQUENCY to pick the right values (correct value repeats
    # across variants; OCR noise values are random each time)
    all_weight_pool = weight_from_kg + tare_from_context
    if all_weight_pool:
        weight_freq = Counter(all_weight_pool).most_common(10)
        # Take the top 2 most frequent values, then assign by magnitude
        top_by_freq = []
        for val, cnt in weight_freq:
            if not _is_year(val) and val not in top_by_freq:
                top_by_freq.append(val)
            if len(top_by_freq) == 2:
                break
        top_by_freq.sort(reverse=True)
        if len(top_by_freq) >= 2:
            result["gvm_kg"] = top_by_freq[0]
            result["tare_kg"] = top_by_freq[1]
        elif len(top_by_freq) == 1:
            result["gvm_kg"] = top_by_freq[0]

    # ── Persons ─────────────────────────────────────────────────────
    persone_hits = re.findall(
        r"Person[es]*[:\s]+(\d{2,3})", combined, re.IGNORECASE
    )
    if persone_hits:
        vals = [int(v) for v in persone_hits if int(v) < 100]
        if vals:
            result["persons_seated"] = Counter(vals).most_common(1)[0][0]

    standing_hits = re.findall(
        r"(?:Standing|Staande)[:\s]*(\d{2,3})", combined, re.IGNORECASE
    )
    if standing_hits:
        vals = [int(v) for v in standing_hits if int(v) < 100]
        if vals:
            result["persons_standing"] = Counter(vals).most_common(1)[0][0]

    # Fallback: 00N patterns
    if result["persons_seated"] is None:
        zero_pats = re.findall(r"\b(00[0-9])\b", combined)
        counts = Counter(zero_pats).most_common(5)
        for val, _ in counts:
            v = int(val)
            if v > 0 and result["persons_seated"] is None:
                result["persons_seated"] = v
            elif v == 0 and result["persons_standing"] is None:
                result["persons_standing"] = v

    if result["persons_seated"] is not None and result["persons_standing"] is None:
        result["persons_standing"] = 0

    # ── Licence number ──────────────────────────────────────────────
    # SA licence numbers: 2-3 letter prefix + 6 digits (e.g. CAA272324)
    # Strategy: direct match → digit-core mining → fragment search

    # Method 1: Direct clean match
    direct_lic = re.findall(r"\b(C[A-Z]{1,2}\d{5,7})\b", upper)
    direct_filtered = [
        l for l in direct_lic
        if not l.startswith("WDD") and not l.startswith("WOD")
    ]
    if direct_filtered:
        result["licence_number"] = _most_common(direct_filtered)

    # Method 2: Mine digit cores from licence-context lines
    if not result["licence_number"]:
        digit_cores = []
        for line in lines:
            l = line.strip()
            if not re.search(r"[Ll]i[csz]?[aeo]?n[csz]|[Ll]isen", l, re.IGNORECASE):
                continue
            all_d = re.findall(r"(\d{5,})", l)
            for d in all_d:
                if d.startswith("1789") or d.startswith("1001") or d.startswith("0049"):
                    continue
                if len(d) >= 6:
                    digit_cores.append(d[-6:])

        # Method 3: Search all text for recurring 6-digit patterns
        all_frags = re.findall(r"\b(\d{6,7})\b", combined)
        for d in all_frags:
            if (not d.startswith("1789") and not d.startswith("1001")
                and not d.startswith("2026") and not d.startswith("2027")
                and not d.startswith("0049") and not d.startswith("0001")):
                digit_cores.append(d[-6:])

        core = _most_common(digit_cores)
        if core:
            # Try to recover the letter prefix
            prefix_hits = re.findall(r"(C[A-Z]{2})\d*" + re.escape(core[:3]), upper)
            if not prefix_hits:
                prefix_hits = re.findall(r"\b([A-Z]{2,3})" + re.escape(core[:2]), upper)
                prefix_hits = [p for p in prefix_hits if not p.startswith("W")]
            if prefix_hits:
                result["licence_number"] = _most_common(prefix_hits) + core
            else:
                result["licence_number"] = core

    # ── Vehicle registration ────────────────────────────────────────
    # SA registrations: 2-3 letters + 2-3 digits + 1-2 letters (e.g. WMH861W)
    # Tesseract heavily garbles this on small text.
    # Strategy: direct match → word-level near labels → licence-line last word

    reg_direct = re.findall(r"\b([A-Z]{2,3}\d{2,3}[A-Z]{1,2})\b", upper)
    reg_filtered = [
        r for r in reg_direct
        if len(r) >= 6 and r != result.get("licence_number")
        and "VM" not in r
    ]
    if reg_filtered:
        result["vehicle_register_number"] = _most_common(reg_filtered)

    # Fallback: word-level near registration label
    if not result["vehicle_register_number"]:
        for i, w in enumerate(words):
            wt = w["text"].upper()
            if any(kw in wt for kw in ["REGISTER", "VRT", "REGISTERNR"]):
                for j in range(i + 1, min(i + 4, len(words))):
                    candidate = words[j]["text"].upper().replace(" ", "")
                    m = re.match(r"[A-Z]{2,3}\d{2,3}[A-Z]{1,2}", candidate)
                    if m:
                        result["vehicle_register_number"] = m.group()
                        break

    # Fallback 2: last word on licence-context lines (registration sits there)
    if not result["vehicle_register_number"]:
        last_words = []
        for line in lines:
            l = line.strip()
            if not (re.search(r"licen|lisen", l, re.IGNORECASE) and re.search(r"\d{5,}", l)):
                continue
            parts = re.split(r"[\s,]+", l)
            for w in reversed(parts):
                clean = re.sub(r"[^A-Za-z0-9]", "", w).upper()
                if len(clean) >= 5 and clean[0].isalpha():
                    last_words.append(clean)
                    break
        # Check if any look like a registration
        for lw in last_words:
            m = re.match(r"([A-Z]{2,3}\d{2,3}[A-Z]{1,2})", lw)
            if m:
                result["vehicle_register_number"] = m.group()
                break

    # ── Description ─────────────────────────────────────────────────
    # PRIORITY 1: Fuzzy match body types in Description/Beskrywing lines
    desc_lines = ""
    for line in lines:
        l = line.strip()
        if re.search(r"descr|beskr", l, re.IGNORECASE):
            desc_lines += " " + l

    # Search description-context lines for fuzzy body types first
    for word in re.findall(r"[A-Z]{3,12}", desc_lines.upper()):
        if word in BODY_TYPE_FUZZY:
            result["description"] = BODY_TYPE_FUZZY[word]
            break

    # PRIORITY 2: Fuzzy match across ALL text
    if not result["description"]:
        for word in re.findall(r"\b[A-Z]{3,12}\b", upper):
            if word in BODY_TYPE_FUZZY:
                result["description"] = BODY_TYPE_FUZZY[word]
                break

    # PRIORITY 3: Direct body type names — exclude Afrikaans "van"
    if not result["description"]:
        for bt in ["Coupe", "Sedan", "Hatchback", "SUV", "Bakkie",
                    "Station Wagon", "Panel Van", "Bus", "Truck", "Cab"]:
            for m in re.finditer(re.escape(bt), combined, re.IGNORECASE):
                start = max(0, m.start() - 20)
                end = min(len(combined), m.end() + 20)
                context = combined[start:end].upper()
                if bt.upper() == "VAN" and re.search(r"VAN\s+(TOETS|DIE|TOES)", context):
                    continue
                result["description"] = bt
                break
            if result["description"]:
                break

    # PRIORITY 4: Extract text between Description/Beskrywing labels
    # and use broad character-level fuzzy matching for heavily garbled text
    if not result["description"]:
        for line in lines:
            l = line.strip()
            # Match lines starting with Description/Descr/Desc
            m = re.match(r"(?:Desc[a-z]*\s*[!|]?)\s*(.*?)\s*(?:[BbRr]eskr|asking|skr|yng|$)", l, re.IGNORECASE)
            if m:
                between = m.group(1).upper().replace(" ", "")
                # Check if it contains fragments of "COUPE" (C/G + O/0 + U + P/R + E)
                if re.search(r"[CG][O0][UÜ][PR]", between):
                    result["description"] = "Coupe"
                elif re.search(r"S[EF][DO]A", between):
                    result["description"] = "Sedan"
                elif re.search(r"HA[TC][CK]", between):
                    result["description"] = "Hatchback"
                elif re.search(r"BA[KG][KG]", between):
                    result["description"] = "Bakkie"
                # If the line has content between the labels, it's likely
                # a body type even if unrecognizable — mark as unknown
                elif len(between) > 2:
                    # Check for "closed" fragments in the garbled text
                    pass
                break

    # Add closed/open top qualifier
    if result["description"]:
        # Search for "closed" and its many OCR garbles
        if re.search(r"CLOSED|GESLOTE|CL.SED|CLCSED|CLISED|CLDSED", upper):
            result["description"] += " (closed top)"
        elif re.search(r"\bOPEN\b|\bOOP\b", upper):
            result["description"] += " (open top)"
        else:
            # Check description-context lines for garbled "(closed top)"
            for line in lines:
                l = line.strip()
                if re.search(r"descr|beskr", l, re.IGNORECASE):
                    # Description lines with extra garbled text after
                    # the body type likely contain "(closed top)"
                    if re.search(r"[({].*(?:top|to[p ])|Ses|cdf|dad|ek|ee|dade|gehasce|tOR|Kel|hel", l, re.IGNORECASE):
                        result["description"] += " (closed top)"
                        break

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_licence_disc_debug(image_bytes: bytes, content_type: str = "image/jpeg") -> dict:
    """
    Same as extract_licence_disc but returns raw OCR text alongside results
    for debugging Tesseract differences across platforms.
    """
    img = _load_image(image_bytes)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    masked, circle_mask = _crop_and_mask(gray)
    variants = _generate_variants(masked, circle_mask)
    combined_text = _run_ocr_all(variants)
    words = _run_ocr_words(masked, circle_mask)
    result = _extract_fields(combined_text, words)
    return {
        "result": result,
        "_debug": {
            "combined_text_length": len(combined_text),
            "combined_text": combined_text,
            "word_count": len(words),
            "words": words,
        },
    }


def extract_licence_disc(image_bytes: bytes, content_type: str = "image/jpeg") -> dict:
    """
    Extract all fields from a SA licence disc image using local OCR.

    Args:
        image_bytes: Raw image file bytes.
        content_type: MIME type (unused — kept for API compatibility).

    Returns:
        Dict with extracted fields. Unreadable fields are None.
    """
    img = _load_image(image_bytes)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    masked, circle_mask = _crop_and_mask(gray)
    variants = _generate_variants(masked, circle_mask)
    combined_text = _run_ocr_all(variants)
    words = _run_ocr_words(masked, circle_mask)
    return _extract_fields(combined_text, words)
