#!/usr/bin/env python3
"""
End-to-end test for the SA Licence Disc OCR API (v2).

Usage:
    1. Start the server:     python app.py
    2. Run tests:            python test_api.py <path_to_disc_image>

    This script uses the ADMIN_API_KEY env var to create a test user,
    then uses that user's key to hit /extract.
"""

import os
import sys
import json
import requests

API_URL = os.environ.get("API_URL", "http://localhost:5000")
ADMIN_KEY = os.environ.get("ADMIN_API_KEY", "test-admin-key")


def admin_headers():
    return {"X-API-Key": ADMIN_KEY}


def user_headers(api_key):
    return {"X-API-Key": api_key}


def pretty(data):
    print(json.dumps(data, indent=2))


def test_health():
    print("═══ Health Check ═══")
    r = requests.get(f"{API_URL}/health")
    pretty(r.json())
    print()


def test_no_auth():
    print("═══ Extract without auth (expect 401) ═══")
    r = requests.post(f"{API_URL}/extract")
    print(f"Status: {r.status_code}")
    pretty(r.json())
    print()


def test_create_user():
    print("═══ Create test user (admin) ═══")
    r = requests.post(
        f"{API_URL}/admin/users",
        headers=admin_headers(),
        json={"email": "test@example.com", "name": "Test User", "plan": "basic"},
    )
    print(f"Status: {r.status_code}")
    data = r.json()
    pretty(data)
    print()

    if r.status_code == 201:
        return data["user"]["api_key"]
    elif r.status_code == 409:
        # User exists — we need to get a key. Regen via admin.
        print("User exists. Checking user list to regen key...")
        r2 = requests.get(f"{API_URL}/admin/users", headers=admin_headers())
        users = r2.json().get("users", [])
        for u in users:
            if u["email"] == "test@example.com":
                r3 = requests.post(
                    f"{API_URL}/admin/users/{u['id']}/regenerate-key",
                    headers=admin_headers(),
                )
                return r3.json().get("api_key")
    return None


def test_extract(api_key, image_path):
    print(f"═══ Extract disc: {image_path} ═══")
    with open(image_path, "rb") as f:
        r = requests.post(
            f"{API_URL}/extract",
            headers=user_headers(api_key),
            files={"image": (image_path, f)},
        )
    print(f"Status: {r.status_code}")
    pretty(r.json())
    print()


def test_me(api_key):
    print("═══ Account info (/me) ═══")
    r = requests.get(f"{API_URL}/me", headers=user_headers(api_key))
    pretty(r.json())
    print()


def test_my_transactions(api_key):
    print("═══ My transactions ═══")
    r = requests.get(f"{API_URL}/me/transactions", headers=user_headers(api_key))
    pretty(r.json())
    print()


def test_admin_transactions():
    print("═══ All transactions (admin) ═══")
    r = requests.get(f"{API_URL}/admin/transactions", headers=admin_headers())
    pretty(r.json())
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <image_path>")
        print("  Set ADMIN_API_KEY env var to match your server config.")
        sys.exit(1)

    image_path = sys.argv[1]

    test_health()
    test_no_auth()

    api_key = test_create_user()
    if not api_key:
        print("ERROR: Could not obtain a user API key.")
        sys.exit(1)

    print(f"Using API key: {api_key[:20]}...\n")

    test_extract(api_key, image_path)
    test_me(api_key)
    test_my_transactions(api_key)
    test_admin_transactions()

    print("═══ All tests complete ═══")
