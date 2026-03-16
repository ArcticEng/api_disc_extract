"""
Database layer — users, subscriptions, and transaction logs.
Uses SQLite by default. Swap to PostgreSQL by changing the connection logic.
"""

import os
import sqlite3
import uuid
import hashlib
import secrets
from datetime import datetime, timedelta
from contextlib import contextmanager

DB_PATH = os.environ.get("DATABASE_PATH", "licence_disc.db")

# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

@contextmanager
def get_db():
    """Yield a SQLite connection with row_factory, auto-commit/rollback."""
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


# ---------------------------------------------------------------------------
# Schema / migrations
# ---------------------------------------------------------------------------

def init_db():
    """Create tables if they don't exist."""
    with get_db() as conn:
        conn.executescript("""
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
        """)


# ---------------------------------------------------------------------------
# API key helpers
# ---------------------------------------------------------------------------

PLAN_LIMITS = {
    "free":       10,
    "basic":      100,
    "pro":        1000,
    "enterprise": 100_000,
}


def _hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


def generate_api_key() -> str:
    """Generate a prefixed API key: disc_live_<32 hex chars>"""
    return f"disc_live_{secrets.token_hex(32)}"


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------

def create_user(email: str, name: str, company: str = None, plan: str = "free") -> dict:
    """Create a new user and return their details including the raw API key."""
    user_id = str(uuid.uuid4())
    api_key = generate_api_key()
    api_key_hash = _hash_key(api_key)
    now = datetime.utcnow().isoformat() + "Z"
    monthly_limit = PLAN_LIMITS.get(plan, 10)

    with get_db() as conn:
        conn.execute("""
            INSERT INTO users
                (id, email, name, company, api_key, api_key_hash, plan,
                 is_active, monthly_limit, requests_used,
                 billing_cycle_start, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, 0, ?, ?, ?)
        """, (user_id, email, name, company, api_key, api_key_hash,
              plan, monthly_limit, now, now, now))

    return {
        "id": user_id,
        "email": email,
        "name": name,
        "company": company,
        "plan": plan,
        "api_key": api_key,          # shown ONCE at creation
        "monthly_limit": monthly_limit,
        "created_at": now,
    }


def get_user_by_api_key(api_key: str) -> dict | None:
    """Look up a user by their raw API key (hashed for comparison)."""
    h = _hash_key(api_key)
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE api_key_hash = ?", (h,)
        ).fetchone()
    return dict(row) if row else None


def get_user_by_email(email: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return dict(row) if row else None


def list_users() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, email, name, company, plan, is_active, "
            "monthly_limit, requests_used, billing_cycle_start, created_at "
            "FROM users ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def update_user(user_id: str, **kwargs) -> bool:
    """Update allowed fields on a user. Returns True if a row was changed."""
    allowed = {"name", "company", "plan", "is_active", "monthly_limit"}
    fields = {k: v for k, v in kwargs.items() if k in allowed}
    if not fields:
        return False

    # If plan changed, update monthly_limit automatically
    if "plan" in fields and "monthly_limit" not in fields:
        fields["monthly_limit"] = PLAN_LIMITS.get(fields["plan"], 10)

    fields["updated_at"] = datetime.utcnow().isoformat() + "Z"
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [user_id]

    with get_db() as conn:
        cur = conn.execute(
            f"UPDATE users SET {set_clause} WHERE id = ?", values
        )
    return cur.rowcount > 0


def deactivate_user(user_id: str) -> bool:
    return update_user(user_id, is_active=0)


def activate_user(user_id: str) -> bool:
    return update_user(user_id, is_active=1)


def regenerate_api_key(user_id: str) -> str | None:
    """Issue a new API key, invalidating the old one."""
    new_key = generate_api_key()
    new_hash = _hash_key(new_key)
    now = datetime.utcnow().isoformat() + "Z"
    with get_db() as conn:
        cur = conn.execute(
            "UPDATE users SET api_key = ?, api_key_hash = ?, updated_at = ? WHERE id = ?",
            (new_key, new_hash, now, user_id),
        )
    return new_key if cur.rowcount > 0 else None


def increment_usage(user_id: str):
    """Increment the monthly request counter."""
    now = datetime.utcnow().isoformat() + "Z"
    with get_db() as conn:
        conn.execute(
            "UPDATE users SET requests_used = requests_used + 1, updated_at = ? WHERE id = ?",
            (now, user_id),
        )


def reset_monthly_usage():
    """Reset all users' request counts — call from a cron / scheduler."""
    now = datetime.utcnow().isoformat() + "Z"
    with get_db() as conn:
        conn.execute(
            "UPDATE users SET requests_used = 0, billing_cycle_start = ?, updated_at = ?",
            (now, now),
        )


def check_subscription(user: dict) -> tuple[bool, str]:
    """
    Check if a user can make a request.
    Returns (allowed: bool, reason: str).
    """
    if not user["is_active"]:
        return False, "Account is deactivated. Contact support."

    if user["requests_used"] >= user["monthly_limit"]:
        return False, (
            f"Monthly limit reached ({user['monthly_limit']} requests on "
            f"'{user['plan']}' plan). Upgrade your plan or wait for the next cycle."
        )

    return True, "ok"


# ---------------------------------------------------------------------------
# Transaction log
# ---------------------------------------------------------------------------

def log_transaction(
    user: dict,
    request_ip: str,
    image_size_bytes: int,
    image_type: str,
    status: str,
    extracted_data: dict | None = None,
    error_message: str | None = None,
    duration_ms: int | None = None,
) -> str:
    """Write a row to the transaction log and return the transaction ID."""
    txn_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"

    # Pull quick-reference fields from extracted data
    vehicle_reg = None
    vehicle_make = None
    vehicle_vin = None
    disc_expiry = None
    extracted_json = None

    if extracted_data and not extracted_data.get("_parse_error"):
        extracted_json = __import__("json").dumps(extracted_data)
        vehicle_reg = extracted_data.get("vehicle_register_number")
        vehicle_make = extracted_data.get("make")
        vehicle_vin = extracted_data.get("vin")
        disc_expiry = extracted_data.get("date_of_expiry")

    api_key_prefix = (user.get("api_key") or "")[:16] + "..."

    with get_db() as conn:
        conn.execute("""
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


def get_transactions(
    user_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """Retrieve transaction log entries, optionally filtered by user."""
    with get_db() as conn:
        if user_id:
            rows = conn.execute(
                "SELECT * FROM transaction_log WHERE user_id = ? "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (user_id, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM transaction_log "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
    return [dict(r) for r in rows]


def get_transaction_by_id(txn_id: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM transaction_log WHERE id = ?", (txn_id,)
        ).fetchone()
    return dict(row) if row else None


def get_user_stats(user_id: str) -> dict:
    """Quick usage summary for a user."""
    with get_db() as conn:
        total = conn.execute(
            "SELECT COUNT(*) as c FROM transaction_log WHERE user_id = ?",
            (user_id,),
        ).fetchone()["c"]

        successes = conn.execute(
            "SELECT COUNT(*) as c FROM transaction_log WHERE user_id = ? AND status = 'success'",
            (user_id,),
        ).fetchone()["c"]

        errors = conn.execute(
            "SELECT COUNT(*) as c FROM transaction_log WHERE user_id = ? AND status = 'error'",
            (user_id,),
        ).fetchone()["c"]

        recent = conn.execute(
            "SELECT created_at FROM transaction_log WHERE user_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (user_id,),
        ).fetchone()

    return {
        "total_requests": total,
        "successful": successes,
        "errors": errors,
        "last_request_at": recent["created_at"] if recent else None,
    }
