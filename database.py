"""
Database layer — users, subscriptions, and transaction logs.
Supports PostgreSQL (production/Railway) and SQLite (local dev).
Auto-detects based on DATABASE_URL environment variable.
"""

import os
import json
import uuid
import hashlib
import secrets
from datetime import datetime
from contextlib import contextmanager

# ---------------------------------------------------------------------------
# Auto-detect database: PostgreSQL if DATABASE_URL is set, else SQLite
# ---------------------------------------------------------------------------
DATABASE_URL = os.environ.get("DATABASE_URL")
USE_POSTGRES = bool(DATABASE_URL)

if USE_POSTGRES:
    import psycopg2
    import psycopg2.extras
else:
    import sqlite3

DB_PATH = os.environ.get("DATABASE_PATH", "licence_disc.db")


@contextmanager
def get_db():
    """Yield a database connection. Works with both PostgreSQL and SQLite."""
    if USE_POSTGRES:
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = False
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def _execute(conn, sql, params=None):
    """Execute a query, adapting placeholder style for the active DB."""
    if USE_POSTGRES:
        # Convert ? placeholders to %s for psycopg2
        sql = sql.replace("?", "%s")
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    else:
        cur = conn.cursor()
    cur.execute(sql, params or ())
    return cur


def _fetchone(conn, sql, params=None):
    cur = _execute(conn, sql, params)
    row = cur.fetchone()
    if row is None:
        return None
    if USE_POSTGRES:
        return dict(row)
    else:
        return dict(row)


def _fetchall(conn, sql, params=None):
    cur = _execute(conn, sql, params)
    rows = cur.fetchall()
    if USE_POSTGRES:
        return [dict(r) for r in rows]
    else:
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Schema / migrations
# ---------------------------------------------------------------------------

_PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS payments (
    id              VARCHAR(36) PRIMARY KEY,
    email           VARCHAR(255) NOT NULL,
    name            VARCHAR(255) NOT NULL,
    company         VARCHAR(255),
    plan            VARCHAR(20) NOT NULL,
    amount          VARCHAR(10) NOT NULL,
    monthly_limit   INTEGER NOT NULL,
    status          VARCHAR(20) DEFAULT 'pending',
    payfast_payment_id VARCHAR(100),
    user_id         VARCHAR(36),
    api_key         VARCHAR(255),
    created_at      VARCHAR(30) NOT NULL,
    completed_at    VARCHAR(30)
);

CREATE TABLE IF NOT EXISTS users (
    id              VARCHAR(36) PRIMARY KEY,
    email           VARCHAR(255) NOT NULL UNIQUE,
    name            VARCHAR(255) NOT NULL,
    company         VARCHAR(255),
    api_key         VARCHAR(255) NOT NULL UNIQUE,
    api_key_hash    VARCHAR(64) NOT NULL UNIQUE,
    plan            VARCHAR(20) NOT NULL DEFAULT 'free',
    is_active       INTEGER NOT NULL DEFAULT 1,
    monthly_limit   INTEGER NOT NULL DEFAULT 10,
    requests_used   INTEGER NOT NULL DEFAULT 0,
    billing_cycle_start VARCHAR(30) NOT NULL,
    created_at      VARCHAR(30) NOT NULL,
    updated_at      VARCHAR(30) NOT NULL
);

CREATE TABLE IF NOT EXISTS transaction_log (
    id              VARCHAR(36) PRIMARY KEY,
    user_id         VARCHAR(36) NOT NULL REFERENCES users(id),
    user_email      VARCHAR(255) NOT NULL,
    user_name       VARCHAR(255) NOT NULL,
    api_key_prefix  VARCHAR(20) NOT NULL,
    request_ip      VARCHAR(45),
    image_size_bytes INTEGER,
    image_type      VARCHAR(50),
    status          VARCHAR(10) NOT NULL DEFAULT 'pending',
    error_message   TEXT,
    extracted_data  TEXT,
    vehicle_reg     VARCHAR(20),
    vehicle_make    VARCHAR(50),
    vehicle_vin     VARCHAR(20),
    disc_expiry     VARCHAR(10),
    duration_ms     INTEGER,
    created_at      VARCHAR(30) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_users_api_key_hash
    ON users(api_key_hash);
CREATE INDEX IF NOT EXISTS idx_transaction_log_user_id
    ON transaction_log(user_id);
CREATE INDEX IF NOT EXISTS idx_transaction_log_created_at
    ON transaction_log(created_at);
CREATE INDEX IF NOT EXISTS idx_transaction_log_vehicle_reg
    ON transaction_log(vehicle_reg);
"""

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS payments (
    id              TEXT PRIMARY KEY,
    email           TEXT NOT NULL,
    name            TEXT NOT NULL,
    company         TEXT,
    plan            TEXT NOT NULL,
    amount          TEXT NOT NULL,
    monthly_limit   INTEGER NOT NULL,
    status          TEXT DEFAULT 'pending',
    payfast_payment_id TEXT,
    user_id         TEXT,
    api_key         TEXT,
    created_at      TEXT NOT NULL,
    completed_at    TEXT
);

CREATE TABLE IF NOT EXISTS users (
    id              TEXT PRIMARY KEY,
    email           TEXT NOT NULL UNIQUE,
    name            TEXT NOT NULL,
    company         TEXT,
    api_key         TEXT NOT NULL UNIQUE,
    api_key_hash    TEXT NOT NULL UNIQUE,
    plan            TEXT NOT NULL DEFAULT 'free'
                        CHECK(plan IN ('free','basic','pro','enterprise')),
    is_active       INTEGER NOT NULL DEFAULT 1,
    monthly_limit   INTEGER NOT NULL DEFAULT 10,
    requests_used   INTEGER NOT NULL DEFAULT 0,
    billing_cycle_start TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transaction_log (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    user_email      TEXT NOT NULL,
    user_name       TEXT NOT NULL,
    api_key_prefix  TEXT NOT NULL,
    request_ip      TEXT,
    image_size_bytes INTEGER,
    image_type      TEXT,
    status          TEXT NOT NULL DEFAULT 'pending'
                        CHECK(status IN ('pending','success','error')),
    error_message   TEXT,
    extracted_data  TEXT,
    vehicle_reg     TEXT,
    vehicle_make    TEXT,
    vehicle_vin     TEXT,
    disc_expiry     TEXT,
    duration_ms     INTEGER,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_users_api_key_hash
    ON users(api_key_hash);
CREATE INDEX IF NOT EXISTS idx_transaction_log_user_id
    ON transaction_log(user_id);
CREATE INDEX IF NOT EXISTS idx_transaction_log_created_at
    ON transaction_log(created_at);
CREATE INDEX IF NOT EXISTS idx_transaction_log_vehicle_reg
    ON transaction_log(vehicle_reg);
"""


def init_db():
    """Create tables if they don't exist."""
    with get_db() as conn:
        if USE_POSTGRES:
            cur = conn.cursor()
            cur.execute(_PG_SCHEMA)
        else:
            conn.executescript(_SQLITE_SCHEMA)


# ---------------------------------------------------------------------------
# API key helpers
# ---------------------------------------------------------------------------

PLAN_LIMITS = {
    "free":       10,
    "basic":      100,
    "pro":        1000,
    "enterprise": 100_000,
    "admin":      999_999_999,
}


def _hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


def generate_api_key() -> str:
    return f"disc_live_{secrets.token_hex(32)}"


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------

def create_user(email: str, name: str, company: str = None, plan: str = "free") -> dict:
    user_id = str(uuid.uuid4())
    api_key = generate_api_key()
    api_key_hash = _hash_key(api_key)
    now = datetime.utcnow().isoformat() + "Z"
    monthly_limit = PLAN_LIMITS.get(plan, 10)

    with get_db() as conn:
        _execute(conn, """
            INSERT INTO users
                (id, email, name, company, api_key, api_key_hash, plan,
                 is_active, monthly_limit, requests_used,
                 billing_cycle_start, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, 0, ?, ?, ?)
        """, (user_id, email, name, company, api_key, api_key_hash,
              plan, monthly_limit, now, now, now))

    return {
        "id": user_id, "email": email, "name": name,
        "company": company, "plan": plan,
        "api_key": api_key,
        "monthly_limit": monthly_limit, "created_at": now,
    }


def get_user_by_api_key(api_key: str) -> dict | None:
    h = _hash_key(api_key)
    with get_db() as conn:
        return _fetchone(conn, "SELECT * FROM users WHERE api_key_hash = ?", (h,))


def get_user_by_email(email: str) -> dict | None:
    with get_db() as conn:
        return _fetchone(conn, "SELECT * FROM users WHERE email = ?", (email,))


def get_user_by_id(user_id: str) -> dict | None:
    with get_db() as conn:
        return _fetchone(conn, "SELECT * FROM users WHERE id = ?", (user_id,))


def list_users() -> list[dict]:
    with get_db() as conn:
        return _fetchall(conn,
            "SELECT id, email, name, company, plan, is_active, "
            "monthly_limit, requests_used, billing_cycle_start, created_at "
            "FROM users ORDER BY created_at DESC")


def update_user(user_id: str, **kwargs) -> bool:
    allowed = {"name", "company", "plan", "is_active", "monthly_limit"}
    fields = {k: v for k, v in kwargs.items() if k in allowed}
    if not fields:
        return False

    if "plan" in fields and "monthly_limit" not in fields:
        fields["monthly_limit"] = PLAN_LIMITS.get(fields["plan"], 10)

    fields["updated_at"] = datetime.utcnow().isoformat() + "Z"
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [user_id]

    with get_db() as conn:
        cur = _execute(conn, f"UPDATE users SET {set_clause} WHERE id = ?", values)
    return cur.rowcount > 0


def deactivate_user(user_id: str) -> bool:
    return update_user(user_id, is_active=0)


def activate_user(user_id: str) -> bool:
    return update_user(user_id, is_active=1)


def regenerate_api_key(user_id: str) -> str | None:
    new_key = generate_api_key()
    new_hash = _hash_key(new_key)
    now = datetime.utcnow().isoformat() + "Z"
    with get_db() as conn:
        cur = _execute(conn,
            "UPDATE users SET api_key = ?, api_key_hash = ?, updated_at = ? WHERE id = ?",
            (new_key, new_hash, now, user_id))
    return new_key if cur.rowcount > 0 else None


def increment_usage(user_id: str):
    now = datetime.utcnow().isoformat() + "Z"
    with get_db() as conn:
        _execute(conn,
            "UPDATE users SET requests_used = requests_used + 1, updated_at = ? WHERE id = ?",
            (now, user_id))


def reset_monthly_usage():
    now = datetime.utcnow().isoformat() + "Z"
    with get_db() as conn:
        _execute(conn,
            "UPDATE users SET requests_used = 0, billing_cycle_start = ?, updated_at = ?",
            (now, now))


def check_subscription(user: dict) -> tuple[bool, str]:
    if not user["is_active"]:
        return False, "Account is deactivated. Contact support."
    if user["requests_used"] >= user["monthly_limit"]:
        return False, (
            f"Lookup limit reached ({user['monthly_limit']} lookups on your "
            f"'{user['plan']}' plan). Please purchase additional lookups at "
            f"https://onyxdigital.co.za/DiscDecode/#pricing")
    return True, "ok"


# ---------------------------------------------------------------------------
# Transaction log
# ---------------------------------------------------------------------------

def log_transaction(
    user: dict, request_ip: str, image_size_bytes: int,
    image_type: str, status: str,
    extracted_data: dict | None = None,
    error_message: str | None = None,
    duration_ms: int | None = None,
) -> str:
    txn_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"

    vehicle_reg = vehicle_make = vehicle_vin = disc_expiry = extracted_json = None
    if extracted_data and not extracted_data.get("_parse_error"):
        extracted_json = json.dumps(extracted_data)
        vehicle_reg = extracted_data.get("vehicle_register_number")
        vehicle_make = extracted_data.get("make")
        vehicle_vin = extracted_data.get("vin")
        disc_expiry = extracted_data.get("date_of_expiry")

    api_key_prefix = (user.get("api_key") or "")[:16] + "..."

    with get_db() as conn:
        _execute(conn, """
            INSERT INTO transaction_log
                (id, user_id, user_email, user_name, api_key_prefix,
                 request_ip, image_size_bytes, image_type,
                 status, error_message, extracted_data,
                 vehicle_reg, vehicle_make, vehicle_vin, disc_expiry,
                 duration_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            txn_id, user["id"], user["email"], user["name"], api_key_prefix,
            request_ip, image_size_bytes, image_type,
            status, error_message, extracted_json,
            vehicle_reg, vehicle_make, vehicle_vin, disc_expiry,
            duration_ms, now,
        ))
    return txn_id


def get_transactions(user_id: str | None = None, limit: int = 50, offset: int = 0) -> list[dict]:
    with get_db() as conn:
        if user_id:
            return _fetchall(conn,
                "SELECT * FROM transaction_log WHERE user_id = ? "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (user_id, limit, offset))
        else:
            return _fetchall(conn,
                "SELECT * FROM transaction_log "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset))


def get_transaction_by_id(txn_id: str) -> dict | None:
    with get_db() as conn:
        return _fetchone(conn, "SELECT * FROM transaction_log WHERE id = ?", (txn_id,))


# ---------------------------------------------------------------------------
# Payments
# ---------------------------------------------------------------------------

PLAN_PRICING = {
    "starter":    {"amount": "9.00",   "limit": 1,    "name": "Starter"},
    "growth":     {"amount": "49.00",  "limit": 10,   "name": "Growth"},
    "business":   {"amount": "89.00",  "limit": 1000, "name": "Business"},
}


def create_payment(email: str, name: str, plan: str, company: str = None) -> dict:
    """Create a pending payment record."""
    pricing = PLAN_PRICING.get(plan)
    if not pricing:
        raise ValueError(f"Invalid plan: {plan}")

    payment_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"

    with get_db() as conn:
        _execute(conn, """
            INSERT INTO payments (id, email, name, company, plan, amount,
                                 monthly_limit, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
        """, (payment_id, email, name, company, plan, pricing["amount"],
              pricing["limit"], now))

    return {
        "id": payment_id, "email": email, "name": name,
        "plan": plan, "amount": pricing["amount"],
        "monthly_limit": pricing["limit"],
    }


def complete_payment(payment_id: str, payfast_payment_id: str = None) -> dict | None:
    """
    Mark payment as complete. Creates a user account and stores the API key.
    Returns the payment record with api_key, or None if not found.
    """
    with get_db() as conn:
        payment = _fetchone(conn, "SELECT * FROM payments WHERE id = ?", (payment_id,))
        if not payment or payment["status"] == "complete":
            return payment  # already processed or not found

        # Create the user
        user = create_user(
            email=payment["email"],
            name=payment["name"],
            company=payment.get("company"),
            plan=payment["plan"],
        )

        # Update monthly_limit to match the plan pricing
        pricing = PLAN_PRICING.get(payment["plan"], {})
        limit = pricing.get("limit", payment["monthly_limit"])
        _execute(conn, "UPDATE users SET monthly_limit = ? WHERE id = ?",
                 (limit, user["id"]))

        now = datetime.utcnow().isoformat() + "Z"
        _execute(conn, """
            UPDATE payments SET status = 'complete', payfast_payment_id = ?,
                   user_id = ?, api_key = ?, completed_at = ?
            WHERE id = ?
        """, (payfast_payment_id, user["id"], user["api_key"], now, payment_id))

        payment["status"] = "complete"
        payment["api_key"] = user["api_key"]
        payment["user_id"] = user["id"]
        return payment


def get_payment(payment_id: str) -> dict | None:
    with get_db() as conn:
        return _fetchone(conn, "SELECT * FROM payments WHERE id = ?", (payment_id,))


def get_user_stats(user_id: str) -> dict:
    with get_db() as conn:
        total = _fetchone(conn,
            "SELECT COUNT(*) as c FROM transaction_log WHERE user_id = ?",
            (user_id,))["c"]
        successes = _fetchone(conn,
            "SELECT COUNT(*) as c FROM transaction_log WHERE user_id = ? AND status = 'success'",
            (user_id,))["c"]
        errors = _fetchone(conn,
            "SELECT COUNT(*) as c FROM transaction_log WHERE user_id = ? AND status = 'error'",
            (user_id,))["c"]
        recent = _fetchone(conn,
            "SELECT created_at FROM transaction_log WHERE user_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (user_id,))

    return {
        "total_requests": total,
        "successful": successes,
        "errors": errors,
        "last_request_at": recent["created_at"] if recent else None,
    }
