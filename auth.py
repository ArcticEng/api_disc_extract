"""
Authentication middleware.
Validates API key from the Authorization header or X-API-Key header,
checks subscription status and rate limits.
"""

import functools
from flask import request, jsonify, g
from database import get_user_by_api_key, check_subscription


def _extract_api_key() -> str | None:
    """Pull the API key from the request headers."""
    # Check X-API-Key header first (preferred)
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return api_key.strip()

    # Fall back to Authorization: Bearer <key>
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()

    return None


def require_auth(f):
    """
    Decorator that enforces API-key authentication + active subscription.
    On success, sets g.current_user to the user dict.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        api_key = _extract_api_key()

        if not api_key:
            return jsonify({
                "error": "Authentication required.",
                "detail": "Provide your API key via the X-API-Key header or "
                          "Authorization: Bearer <key> header.",
            }), 401

        user = get_user_by_api_key(api_key)

        if user is None:
            return jsonify({
                "error": "Invalid API key.",
            }), 401

        # Subscription / rate-limit check
        allowed, reason = check_subscription(user)
        if not allowed:
            return jsonify({
                "error": "Subscription check failed.",
                "detail": reason,
                "plan": user["plan"],
                "requests_used": user["requests_used"],
                "monthly_limit": user["monthly_limit"],
            }), 403

        # Attach user to request context
        g.current_user = user
        return f(*args, **kwargs)

    return wrapper


def require_admin(f):
    """
    Decorator for admin-only endpoints.
    Checks the ADMIN_API_KEY env variable.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        import os
        admin_key = os.environ.get("ADMIN_API_KEY")
        if not admin_key:
            return jsonify({
                "error": "Admin endpoints are not configured (ADMIN_API_KEY not set)."
            }), 503

        provided = _extract_api_key()
        if provided != admin_key:
            return jsonify({"error": "Unauthorized. Admin access required."}), 403

        return f(*args, **kwargs)

    return wrapper
