"""
Authentication middleware.
Validates API key from the Authorization header or X-API-Key header,
checks subscription status and rate limits.
"""

import functools
import logging
from flask import request, jsonify, g
from database import get_user_by_api_key, check_subscription

logger = logging.getLogger(__name__)


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

        client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)

        if not api_key:
            logger.warning("AUTH_FAIL: No API key provided from IP=%s path=%s",
                          client_ip, request.path)
            return jsonify({
                "error": "Authentication required.",
                "detail": "Provide your API key via the X-API-Key header or "
                          "Authorization: Bearer <key> header.",
            }), 401

        user = get_user_by_api_key(api_key)

        if user is None:
            logger.warning("AUTH_FAIL: Invalid API key prefix=%s... from IP=%s path=%s",
                          api_key[:16], client_ip, request.path)
            return jsonify({
                "error": "Invalid API key.",
            }), 401

        # Subscription / rate-limit check
        allowed, reason = check_subscription(user)
        if not allowed:
            logger.warning("RATE_LIMIT: user=%s plan=%s used=%d/%d from IP=%s",
                          user["email"], user["plan"],
                          user["requests_used"], user["monthly_limit"], client_ip)
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
