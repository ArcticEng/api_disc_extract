"""
South African License Disc OCR API  (v3.0 — Claude Vision)
=============================================================
- Claude Vision AI — handles any rotation, angle, or lighting
- API-key authentication with subscription tiers
- Per-user monthly rate limits
- Full transaction logging (who, when, what, result)
- Admin endpoints for user + log management
"""

import os
import json
import time
import hashlib
import urllib.parse
import logging
from datetime import datetime

from flask import Flask, request, jsonify, g
from flask_cors import CORS

import database as db
from auth import require_auth, require_admin
from ocr_engine import (
    extract_licence_disc,
    extract_drivers_licence, extract_document,
    get_supported_doc_types,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff"}
MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Initialise database on startup
db.init_db()
if db.USE_POSTGRES:
    logger.info("Database initialised: PostgreSQL")
else:
    logger.info("Database initialised: SQLite at %s", db.DB_PATH)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "SA Licence Disc OCR API",
        "version": "3.0.0 — Claude Vision",
        "endpoints": {
            "POST   /extract":            "Extract licence disc data (requires API key)",
            "POST   /extract-licence":    "Extract drivers licence data (requires API key)",
            "POST   /extract-doc":        "Extract ANY document — pass doc_type param (requires API key)",
            "GET    /doc-types":           "List all supported document types",
            "GET    /me":                  "Your account & usage info",
            "GET    /me/transactions":     "Your transaction history",
            "GET    /health":              "Health check",
        },
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "ocr_engine": "claude-vision",
        "anthropic_key_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "database": "postgresql" if db.USE_POSTGRES else "sqlite",
        "admin_configured": bool(os.environ.get("ADMIN_API_KEY")),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


# ═══════════════════════════════════════════════════════════════════════════
# AUTHENTICATED — extraction endpoint
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/extract", methods=["POST"])
@require_auth
def extract():
    """
    Authenticated endpoint: extract licence disc data from an uploaded image.
    Logs the transaction and increments the user's usage counter.
    """
    user = g.current_user
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    # ── Parse image from request ──────────────────────────────────────────
    if "image" in request.files:
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}"}), 400
        image_bytes = file.read()
        content_type = file.content_type or "image/jpeg"
    elif request.data:
        image_bytes = request.data
        content_type = request.content_type or "image/jpeg"
    else:
        return jsonify({"error": "No image provided. Send as multipart 'image' field or raw body."}), 400

    if len(image_bytes) < 1000:
        return jsonify({"error": "Image too small — likely not a valid photo."}), 400

    logger.info(
        "User %s (%s) — extracting disc  size=%d  type=%s",
        user["email"], user["plan"], len(image_bytes), content_type,
    )

    # ── Run local OCR extraction ─────────────────────────────────────────
    start_ms = time.time()
    try:
        result = extract_licence_disc(image_bytes, content_type)
        duration_ms = int((time.time() - start_ms) * 1000)

        # Log success
        txn_id = db.log_transaction(
            user=user,
            request_ip=client_ip,
            image_size_bytes=len(image_bytes),
            image_type=content_type,
            status="success",
            extracted_data=result,
            duration_ms=duration_ms,
            doc_type="licence_disc",
        )
        db.increment_usage(user["id"])

        return jsonify({
            "success": True,
            "transaction_id": txn_id,
            "data": result,
            "usage": {
                "requests_used": user["requests_used"] + 1,
                "monthly_limit": user["monthly_limit"],
                "plan": user["plan"],
            },
        })

    except Exception as exc:
        duration_ms = int((time.time() - start_ms) * 1000)
        db.log_transaction(
            user=user, request_ip=client_ip,
            image_size_bytes=len(image_bytes), image_type=content_type,
            status="error", error_message=str(exc), duration_ms=duration_ms,
            doc_type="licence_disc",
        )
        logger.exception("Unexpected error")
        return jsonify({"error": str(exc)}), 500


# ═══════════════════════════════════════════════════════════════════════════
# AUTHENTICATED — drivers licence extraction
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/extract-licence", methods=["POST"])
@require_auth
def extract_licence():
    """
    Extract data from a SA driving licence card.
    Same auth and logging as /extract.
    """
    user = g.current_user
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    if "image" in request.files:
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}"}), 400
        image_bytes = file.read()
        content_type = file.content_type or "image/jpeg"
    elif request.data:
        image_bytes = request.data
        content_type = request.content_type or "image/jpeg"
    else:
        return jsonify({"error": "No image provided."}), 400

    if len(image_bytes) < 1000:
        return jsonify({"error": "Image too small."}), 400

    logger.info("User %s — extracting drivers licence  size=%d", user["email"], len(image_bytes))

    start_ms = time.time()
    try:
        result = extract_drivers_licence(image_bytes, content_type)
        duration_ms = int((time.time() - start_ms) * 1000)

        txn_id = db.log_transaction(
            user=user, request_ip=client_ip,
            image_size_bytes=len(image_bytes), image_type=content_type,
            status="success", extracted_data=result, duration_ms=duration_ms,
            doc_type="drivers_licence",
        )
        db.increment_usage(user["id"])

        return jsonify({
            "success": True,
            "type": "drivers_licence",
            "transaction_id": txn_id,
            "data": result,
            "usage": {
                "requests_used": user["requests_used"] + 1,
                "monthly_limit": user["monthly_limit"],
                "plan": user["plan"],
            },
        })

    except Exception as exc:
        duration_ms = int((time.time() - start_ms) * 1000)
        db.log_transaction(
            user=user, request_ip=client_ip,
            image_size_bytes=len(image_bytes), image_type=content_type,
            status="error", error_message=str(exc), duration_ms=duration_ms,
            doc_type="drivers_licence",
        )
        logger.exception("Error extracting drivers licence")
        return jsonify({"error": str(exc)}), 500


# ═══════════════════════════════════════════════════════════════════════════
# GENERIC document extraction (any document type)
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/doc-types", methods=["GET"])
def list_doc_types():
    """List all supported document types."""
    return jsonify(get_supported_doc_types())


@app.route("/extract-doc", methods=["POST"])
@require_auth
def extract_doc():
    """
    Extract data from ANY document type.
    Pass doc_type as form field or query param.
    Supported types: licence_disc, drivers_licence, id_document,
                     vehicle_registration, invoice, generic
    """
    user = g.current_user
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    # Get doc_type from form data, query param, or JSON
    doc_type = (request.form.get("doc_type")
                or request.args.get("doc_type")
                or "generic")

    supported = get_supported_doc_types()
    if doc_type not in supported:
        return jsonify({
            "error": f"Unknown doc_type: '{doc_type}'",
            "supported_types": supported,
        }), 400

    if "image" in request.files:
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}"}), 400
        image_bytes = file.read()
        content_type = file.content_type or "image/jpeg"
    elif request.data:
        image_bytes = request.data
        content_type = request.content_type or "image/jpeg"
    else:
        return jsonify({"error": "No image provided."}), 400

    if len(image_bytes) < 1000:
        return jsonify({"error": "Image too small."}), 400

    logger.info("User %s — extracting %s  size=%d", user["email"], doc_type, len(image_bytes))

    start_ms = time.time()
    try:
        result = extract_document(image_bytes, doc_type, content_type)
        duration_ms = int((time.time() - start_ms) * 1000)

        txn_id = db.log_transaction(
            user=user, request_ip=client_ip,
            image_size_bytes=len(image_bytes), image_type=content_type,
            status="success", extracted_data=result, duration_ms=duration_ms,
            doc_type=doc_type,
        )
        db.increment_usage(user["id"])

        return jsonify({
            "success": True,
            "type": doc_type,
            "transaction_id": txn_id,
            "data": result,
            "usage": {
                "requests_used": user["requests_used"] + 1,
                "monthly_limit": user["monthly_limit"],
                "plan": user["plan"],
            },
        })

    except Exception as exc:
        duration_ms = int((time.time() - start_ms) * 1000)
        db.log_transaction(
            user=user, request_ip=client_ip,
            image_size_bytes=len(image_bytes), image_type=content_type,
            status="error", error_message=str(exc), duration_ms=duration_ms,
            doc_type=doc_type,
        )
        logger.exception("Error extracting %s", doc_type)
        return jsonify({"error": str(exc)}), 500


# ═══════════════════════════════════════════════════════════════════════════
# AUTHENTICATED — self-service account info
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/me", methods=["GET"])
@require_auth
def me():
    """Return the authenticated user's account details and usage stats."""
    user = g.current_user
    stats = db.get_user_stats(user["id"])
    return jsonify({
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "company": user["company"],
            "plan": user["plan"],
            "is_active": bool(user["is_active"]),
            "monthly_limit": user["monthly_limit"],
            "requests_used": user["requests_used"],
            "billing_cycle_start": user["billing_cycle_start"],
            "created_at": user["created_at"],
        },
        "stats": stats,
    })


@app.route("/me/transactions", methods=["GET"])
@require_auth
def my_transactions():
    """Return the authenticated user's transaction history."""
    user = g.current_user
    limit = min(int(request.args.get("limit", 50)), 200)
    offset = int(request.args.get("offset", 0))
    txns = db.get_transactions(user_id=user["id"], limit=limit, offset=offset)

    # Redact raw extracted_data in list view for brevity
    for t in txns:
        if t.get("extracted_data"):
            t["extracted_data"] = "(available in detail view)"

    return jsonify({"transactions": txns, "count": len(txns), "limit": limit, "offset": offset})


# ═══════════════════════════════════════════════════════════════════════════
# ADMIN ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/admin/users", methods=["POST"])
@require_admin
def admin_create_user():
    """Create a new user account. Returns the API key (shown once)."""
    body = request.get_json(force=True)
    email = body.get("email")
    name = body.get("name")
    company = body.get("company")
    plan = body.get("plan", "free")

    if not email or not name:
        return jsonify({"error": "email and name are required."}), 400

    if plan not in db.PLAN_LIMITS:
        return jsonify({"error": f"Invalid plan. Choose from: {list(db.PLAN_LIMITS.keys())}"}), 400

    if db.get_user_by_email(email):
        return jsonify({"error": f"A user with email '{email}' already exists."}), 409

    user = db.create_user(email=email, name=name, company=company, plan=plan)
    logger.info("Admin created user %s (%s) on plan=%s", email, user["id"], plan)

    return jsonify({
        "message": "User created. The API key below is shown ONCE — store it securely.",
        "user": user,
    }), 201


@app.route("/admin/users", methods=["GET"])
@require_admin
def admin_list_users():
    users = db.list_users()
    return jsonify({"users": users, "count": len(users)})


@app.route("/admin/users/<user_id>", methods=["GET"])
@require_admin
def admin_get_user(user_id):
    user = db.get_user_by_id(user_id)
    if not user:
        return jsonify({"error": "User not found."}), 404
    stats = db.get_user_stats(user_id)
    user.pop("api_key", None)
    user.pop("api_key_hash", None)
    return jsonify({"user": user, "stats": stats})


@app.route("/admin/users/<user_id>", methods=["PATCH"])
@require_admin
def admin_update_user(user_id):
    """Update user fields: name, company, plan, is_active, monthly_limit."""
    body = request.get_json(force=True)
    if not db.get_user_by_id(user_id):
        return jsonify({"error": "User not found."}), 404

    updated = db.update_user(user_id, **body)
    if not updated:
        return jsonify({"error": "No valid fields to update."}), 400

    return jsonify({"message": "User updated.", "user": db.get_user_by_id(user_id)})


@app.route("/admin/users/<user_id>/deactivate", methods=["POST"])
@require_admin
def admin_deactivate_user(user_id):
    if db.deactivate_user(user_id):
        return jsonify({"message": "User deactivated."})
    return jsonify({"error": "User not found."}), 404


@app.route("/admin/users/<user_id>/activate", methods=["POST"])
@require_admin
def admin_activate_user(user_id):
    if db.activate_user(user_id):
        return jsonify({"message": "User activated."})
    return jsonify({"error": "User not found."}), 404


@app.route("/admin/users/<user_id>/regenerate-key", methods=["POST"])
@require_admin
def admin_regenerate_key(user_id):
    new_key = db.regenerate_api_key(user_id)
    if new_key:
        return jsonify({
            "message": "API key regenerated. The old key is now invalid.",
            "api_key": new_key,
        })
    return jsonify({"error": "User not found."}), 404


@app.route("/admin/transactions", methods=["GET"])
@require_admin
def admin_transactions():
    """Browse all transaction logs. Filter by ?user_id= optionally."""
    user_id = request.args.get("user_id")
    limit = min(int(request.args.get("limit", 50)), 500)
    offset = int(request.args.get("offset", 0))
    txns = db.get_transactions(user_id=user_id, limit=limit, offset=offset)
    return jsonify({"transactions": txns, "count": len(txns), "limit": limit, "offset": offset})


@app.route("/admin/transactions/<txn_id>", methods=["GET"])
@require_admin
def admin_transaction_detail(txn_id):
    txn = db.get_transaction_by_id(txn_id)
    if not txn:
        return jsonify({"error": "Transaction not found."}), 404
    if txn.get("extracted_data"):
        try:
            txn["extracted_data"] = json.loads(txn["extracted_data"])
        except json.JSONDecodeError:
            pass
    return jsonify({"transaction": txn})


@app.route("/admin/reset-usage", methods=["POST"])
@require_admin
def admin_reset_usage():
    """Reset all users' monthly request counters."""
    db.reset_monthly_usage()
    logger.info("Admin reset monthly usage counters for all users.")
    return jsonify({"message": "Monthly usage counters reset for all users."})


# ═══════════════════════════════════════════════════════════════════════════
# PAYMENT / PAYFAST ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

PAYFAST_MERCHANT_ID = os.environ.get("PAYFAST_MERCHANT_ID")
PAYFAST_MERCHANT_KEY = os.environ.get("PAYFAST_MERCHANT_KEY")
PAYFAST_PASSPHRASE = os.environ.get("PAYFAST_PASSPHRASE", "")
PAYFAST_SANDBOX = os.environ.get("PAYFAST_SANDBOX", "false").lower() == "true"
PAYFAST_URL = "https://sandbox.payfast.co.za/eng/process" if PAYFAST_SANDBOX else "https://www.payfast.co.za/eng/process"
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:3000")


def _payfast_signature(data: dict) -> str:
    """Generate PayFast MD5 signature."""
    # PayFast requires: fields in submission order, empty values excluded,
    # values URL-encoded, then MD5 hashed.
    pf_str = "&".join(
        f"{k}={urllib.parse.quote_plus(str(v).strip())}"
        for k, v in data.items()
        if v is not None and str(v).strip() != ""
    )
    if PAYFAST_PASSPHRASE:
        pf_str += f"&passphrase={urllib.parse.quote_plus(PAYFAST_PASSPHRASE.strip())}"
    return hashlib.md5(pf_str.encode()).hexdigest()


@app.route("/api/create-payment", methods=["POST"])
def create_payment():
    """
    Create a pending payment and return PayFast form fields.
    Frontend uses these to redirect user to PayFast.
    """
    body = request.get_json(force=True)
    email = body.get("email", "").strip()
    name = body.get("name", "").strip()
    plan = body.get("plan", "").strip()
    company = body.get("company", "").strip() or None

    if not email or not name or not plan:
        return jsonify({"error": "email, name, and plan are required."}), 400

    if plan not in db.PLAN_PRICING:
        return jsonify({"error": f"Invalid plan. Choose from: {list(db.PLAN_PRICING.keys())}"}), 400

    # Check if email already exists
    existing = db.get_user_by_email(email)
    if existing:
        return jsonify({"error": "An account with this email already exists. Please use a different email."}), 409

    try:
        payment = db.create_payment(email=email, name=name, plan=plan, company=company)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    pricing = db.PLAN_PRICING[plan]

    # Build PayFast form data
    # Build PayFast data — order matters for signature!
    # Empty values must be excluded entirely.
    name_parts = name.split() if name else [name]
    name_first = name_parts[0]
    name_last = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""

    pf_data = {}
    pf_data["merchant_id"] = PAYFAST_MERCHANT_ID
    pf_data["merchant_key"] = PAYFAST_MERCHANT_KEY
    pf_data["return_url"] = f"{FRONTEND_URL}/success.html?ref={payment['id']}"
    pf_data["cancel_url"] = f"{FRONTEND_URL}/#pricing"
    # Force HTTPS for notify_url (Railway proxy reports HTTP)
    base_url = request.url_root.rstrip('/').replace('http://', 'https://')
    pf_data["notify_url"] = f"{base_url}/webhook/payfast"
    if name_first:
        pf_data["name_first"] = name_first
    if name_last:
        pf_data["name_last"] = name_last
    pf_data["email_address"] = email
    pf_data["m_payment_id"] = payment["id"]
    pf_data["amount"] = pricing["amount"]
    pf_data["item_name"] = f"DiscDecode {pricing['name']} Plan"

    pf_data["signature"] = _payfast_signature(pf_data)

    return jsonify({
        "payment_id": payment["id"],
        "payfast_url": PAYFAST_URL,
        "payfast_data": pf_data,
    })


@app.route("/webhook/payfast", methods=["POST"])
def payfast_webhook():
    """Receive PayFast ITN (Instant Transaction Notification)."""
    data = request.form.to_dict()
    logger.info("PayFast ITN received: m_payment_id=%s status=%s",
                data.get("m_payment_id"), data.get("payment_status"))

    payment_id = data.get("m_payment_id")
    payment_status = data.get("payment_status")
    pf_payment_id = data.get("pf_payment_id")

    if not payment_id:
        return "missing m_payment_id", 400

    # Only process COMPLETE payments
    if payment_status != "COMPLETE":
        logger.info("PayFast ITN: status=%s (not COMPLETE), skipping.", payment_status)
        return "ok", 200

    try:
        result = db.complete_payment(payment_id, payfast_payment_id=pf_payment_id)
        if result:
            logger.info("Payment %s completed. User created: %s", payment_id, result.get("email"))
        else:
            logger.warning("Payment %s not found or already processed.", payment_id)
    except Exception as exc:
        logger.exception("Error processing PayFast ITN for %s", payment_id)
        return "error", 500

    return "ok", 200


@app.route("/api/payment-status/<payment_id>", methods=["GET"])
def payment_status(payment_id):
    """Check payment status. Returns API key if payment is complete."""
    payment = db.get_payment(payment_id)
    if not payment:
        return jsonify({"error": "Payment not found."}), 404

    result = {
        "status": payment["status"],
        "plan": payment["plan"],
        "email": payment["email"],
    }

    if payment["status"] == "complete":
        result["api_key"] = payment["api_key"]
        result["monthly_limit"] = payment["monthly_limit"]

    return jsonify(result)


# ═══════════════════════════════════════════════════════════════════════════
# Entry-point
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
