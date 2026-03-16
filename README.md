# SA Licence Disc OCR API  v2.1 — Local OCR

A REST API that extracts all fields from a **South African vehicle licence disc** photo using local Tesseract OCR — no external API calls, zero per-request cost.

---

## How it works

The OCR engine uses a multi-pass approach to maximise accuracy:

1. **Disc detection** — finds the white circular disc on any background using contour detection
2. **Circle masking** — crops to the disc and whites-out the surrounding area
3. **8 preprocessing variants** — adaptive thresholding with different block sizes and C-values, across both raw and bilateral-filtered images
4. **Tesseract OCR** — runs on every variant and combines all text
5. **Regex pattern mining** — extracts each field using patterns calibrated to SA disc layouts, with frequency voting across variants for accuracy

## Extracted Fields

| Field                    | Example Value              | Accuracy |
|--------------------------|----------------------------|----------|
| `disc_number`            | `1001056VM2N1`             | High     |
| `licence_number`         | `272324` (digits only*)    | Medium   |
| `vehicle_register_number`| `WMH861W`                  | Low*     |
| `vin`                    | `WDD1903772A004905`        | High     |
| `engine_number`          | `17898060004192`           | High     |
| `make`                   | `MERCEDES-BENZ`            | High     |
| `description`            | `Coupe`                    | Medium   |
| `fees`                   | `2.2`                      | High     |
| `gvm_kg`                 | `1890`                     | High     |
| `tare_kg`                | `1615`                     | High     |
| `date_of_test`           | `2026-01-26`               | High     |
| `persons_seated`         | `2`                        | High     |
| `persons_standing`       | `0`                        | High     |
| `date_of_expiry`         | `2027-02-28`               | High     |
| `country`                | `RSA`                      | High     |

*\*Licence number and registration are printed in small text on the disc — Tesseract struggles with these. The digit portion is usually captured but the letter prefix may be garbled or missing. Better camera quality and lighting significantly improve results.*

---

## Quick Start

### 1. Install Tesseract (system dependency)

```bash
# Ubuntu / Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Verify
tesseract --version
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Create your first user

```bash
export ADMIN_API_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
python manage.py create-user --email you@company.com --name "Your Name" --plan pro
```

Save the API key printed — it's shown only once.

### 4. Start the server

```bash
python app.py
```

### 5. Extract a disc

```bash
curl -X POST http://localhost:5000/extract \
  -H "X-API-Key: disc_live_..." \
  -F "image=@disc.jpeg"
```

---

## API Reference

### Public

| Method | Endpoint   | Description       |
|--------|------------|-------------------|
| GET    | `/`        | Service info      |
| GET    | `/health`  | Health check      |

### Authenticated (user API key via `X-API-Key` header)

| Method | Endpoint            | Description                    |
|--------|---------------------|--------------------------------|
| POST   | `/extract`          | Extract disc data from image   |
| GET    | `/me`               | Your account info + usage      |
| GET    | `/me/transactions`  | Your transaction history       |

### Admin (`ADMIN_API_KEY` via `X-API-Key` header)

| Method | Endpoint                              | Description                |
|--------|---------------------------------------|----------------------------|
| POST   | `/admin/users`                        | Create a user              |
| GET    | `/admin/users`                        | List all users             |
| GET    | `/admin/users/:id`                    | Get user details + stats   |
| PATCH  | `/admin/users/:id`                    | Update user fields         |
| POST   | `/admin/users/:id/deactivate`         | Suspend a user             |
| POST   | `/admin/users/:id/activate`           | Reactivate a user          |
| POST   | `/admin/users/:id/regenerate-key`     | Issue new API key          |
| GET    | `/admin/transactions`                 | Browse all logs            |
| GET    | `/admin/transactions/:id`             | Full transaction detail    |
| POST   | `/admin/reset-usage`                  | Reset monthly counters     |

### Subscription Plans

| Plan       | Monthly Limit |
|------------|---------------|
| free       | 10            |
| basic      | 100           |
| pro        | 1,000         |
| enterprise | 100,000       |

### Example Response (`POST /extract`)

```json
{
  "success": true,
  "transaction_id": "a1b2c3d4-...",
  "data": {
    "disc_number": "1001056VM2N1",
    "licence_number": "272324",
    "vehicle_register_number": null,
    "vin": "WDD1903772A004905",
    "engine_number": "17898060004192",
    "make": "MERCEDES-BENZ",
    "description": "Coupe",
    "fees": 2.2,
    "gvm_kg": 1890,
    "tare_kg": 1615,
    "date_of_test": "2026-01-26",
    "persons_seated": 2,
    "persons_standing": 0,
    "date_of_expiry": "2027-02-28",
    "country": "RSA"
  },
  "usage": {
    "requests_used": 4,
    "monthly_limit": 1000,
    "plan": "pro"
  }
}
```

---

## CLI Management (`manage.py`)

```bash
python manage.py create-user  --email user@co.za --name "Jan" --plan pro
python manage.py list-users
python manage.py user-info    --email user@co.za
python manage.py deactivate   --email user@co.za
python manage.py activate     --email user@co.za
python manage.py regen-key    --email user@co.za
python manage.py transactions --limit 20
python manage.py user-txns    --email user@co.za
python manage.py reset-usage
python manage.py seed-demo
```

---

## Docker

```bash
docker build -t sa-disc-api .
docker run -p 5000:5000 \
  -e ADMIN_API_KEY="your-admin-secret" \
  -v ./data:/app/data \
  sa-disc-api
```

---

## Tips for Best Results

- **Lighting**: even, diffused light with no glare on the disc
- **Angle**: photograph straight-on, not at an angle
- **Focus**: make sure text is sharp, not blurry
- **Resolution**: at least 800×800 pixels covering the disc
- **Background**: dark backgrounds help disc detection

---

## Architecture

```
Client request
      │
      ▼
  Auth middleware ──→ 401/403 if invalid key or limit reached
      │
      ▼
  POST /extract
      │
      ├─→ Disc detection (OpenCV contour detection)
      ├─→ Circle crop + mask
      ├─→ 8 preprocessing variants (adaptive threshold)
      ├─→ Tesseract OCR × 8 runs
      ├─→ Combined text → regex field mining
      │
      ▼
  Log transaction ──→ SQLite (users + transaction_log)
      │
      ▼
  JSON response
```

## Tech Stack

- **Flask** + **Flask-CORS** — API framework
- **Tesseract OCR** — on-device text recognition
- **OpenCV** — image preprocessing pipeline
- **SQLite** — user accounts + transaction log
- **Gunicorn** — production WSGI server
