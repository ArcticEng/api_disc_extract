#!/usr/bin/env python3
"""
manage.py — CLI for managing users, API keys, and transaction logs.

Usage:
    python manage.py create-user  --email user@co.za --name "John" --plan pro
    python manage.py list-users
    python manage.py user-info    --email user@co.za
    python manage.py deactivate   --email user@co.za
    python manage.py activate     --email user@co.za
    python manage.py regen-key    --email user@co.za
    python manage.py transactions --limit 20
    python manage.py user-txns    --email user@co.za
    python manage.py reset-usage
    python manage.py seed-demo
"""

import argparse
import json
import sys

import database as db


def cmd_create_user(args):
    db.init_db()
    if db.get_user_by_email(args.email):
        print(f"ERROR: User with email '{args.email}' already exists.")
        sys.exit(1)

    user = db.create_user(
        email=args.email,
        name=args.name,
        company=args.company,
        plan=args.plan,
    )
    print("\n  User created successfully!")
    print(f"  ID:        {user['id']}")
    print(f"  Email:     {user['email']}")
    print(f"  Name:      {user['name']}")
    print(f"  Plan:      {user['plan']} ({user['monthly_limit']} requests/month)")
    print(f"\n  API Key:   {user['api_key']}")
    print("\n  *** Save this API key — it is shown only once! ***\n")


def cmd_list_users(args):
    db.init_db()
    users = db.list_users()
    if not users:
        print("No users found.")
        return

    print(f"\n{'Email':<30} {'Name':<20} {'Plan':<12} {'Used':<8} {'Limit':<8} {'Active'}")
    print("-" * 110)
    for u in users:
        active = "Yes" if u["is_active"] else "NO"
        print(f"{u['email']:<30} {u['name']:<20} {u['plan']:<12} {u['requests_used']:<8} {u['monthly_limit']:<8} {active}")
    print(f"\nTotal: {len(users)} users\n")


def cmd_user_info(args):
    db.init_db()
    user = db.get_user_by_email(args.email)
    if not user:
        print(f"User '{args.email}' not found.")
        sys.exit(1)

    stats = db.get_user_stats(user["id"])
    print(f"\n  ID:              {user['id']}")
    print(f"  Email:           {user['email']}")
    print(f"  Name:            {user['name']}")
    print(f"  Company:         {user['company'] or '—'}")
    print(f"  Plan:            {user['plan']}")
    print(f"  Active:          {'Yes' if user['is_active'] else 'NO'}")
    print(f"  Requests used:   {user['requests_used']} / {user['monthly_limit']}")
    print(f"  Total all-time:  {stats['total_requests']}")
    print(f"  Successful:      {stats['successful']}")
    print(f"  Errors:          {stats['errors']}")
    print(f"  Last request:    {stats['last_request_at'] or '—'}")
    print(f"  Created:         {user['created_at']}\n")


def cmd_deactivate(args):
    db.init_db()
    user = db.get_user_by_email(args.email)
    if not user:
        print(f"User '{args.email}' not found.")
        sys.exit(1)
    db.deactivate_user(user["id"])
    print(f"User '{args.email}' has been deactivated.")


def cmd_activate(args):
    db.init_db()
    user = db.get_user_by_email(args.email)
    if not user:
        print(f"User '{args.email}' not found.")
        sys.exit(1)
    db.activate_user(user["id"])
    print(f"User '{args.email}' has been activated.")


def cmd_regen_key(args):
    db.init_db()
    user = db.get_user_by_email(args.email)
    if not user:
        print(f"User '{args.email}' not found.")
        sys.exit(1)
    new_key = db.regenerate_api_key(user["id"])
    print(f"\n  New API Key: {new_key}")
    print("  *** The old key is now invalid. Save this one! ***\n")


def cmd_transactions(args):
    db.init_db()
    txns = db.get_transactions(limit=args.limit, offset=args.offset)
    if not txns:
        print("No transactions found.")
        return

    print(f"\n{'Date':<22} {'User':<25} {'Status':<10} {'Reg':<12} {'Make':<18} {'Duration'}")
    print("-" * 110)
    for t in txns:
        date = t["created_at"][:19]
        reg = t["vehicle_reg"] or "—"
        make = t["vehicle_make"] or "—"
        dur = f"{t['duration_ms']}ms" if t["duration_ms"] else "—"
        print(f"{date:<22} {t['user_email']:<25} {t['status']:<10} {reg:<12} {make:<18} {dur}")
    print(f"\nShowing {len(txns)} transactions\n")


def cmd_user_txns(args):
    db.init_db()
    user = db.get_user_by_email(args.email)
    if not user:
        print(f"User '{args.email}' not found.")
        sys.exit(1)
    txns = db.get_transactions(user_id=user["id"], limit=args.limit)
    if not txns:
        print(f"No transactions for '{args.email}'.")
        return

    print(f"\nTransactions for {args.email}:")
    print(f"{'Date':<22} {'Status':<10} {'Reg':<12} {'Make':<18} {'VIN':<20} {'Duration'}")
    print("-" * 100)
    for t in txns:
        date = t["created_at"][:19]
        reg = t["vehicle_reg"] or "—"
        make = t["vehicle_make"] or "—"
        vin = t["vehicle_vin"] or "—"
        dur = f"{t['duration_ms']}ms" if t["duration_ms"] else "—"
        print(f"{date:<22} {t['status']:<10} {reg:<12} {make:<18} {vin:<20} {dur}")
    print()


def cmd_reset_usage(args):
    db.init_db()
    db.reset_monthly_usage()
    print("Monthly usage counters reset for all users.")


def cmd_seed_demo(args):
    """Create demo users for testing."""
    db.init_db()
    demos = [
        {"email": "demo@example.com",    "name": "Demo User",      "plan": "free"},
        {"email": "pro@example.com",     "name": "Pro Tester",     "plan": "pro",   "company": "TestCo"},
        {"email": "enterprise@acme.com", "name": "Enterprise Bot", "plan": "enterprise", "company": "Acme Corp"},
    ]
    print("\nCreating demo users:\n")
    for d in demos:
        if db.get_user_by_email(d["email"]):
            print(f"  SKIP  {d['email']} (already exists)")
            continue
        user = db.create_user(**d)
        print(f"  OK    {d['email']:<30} plan={d['plan']:<12} key={user['api_key']}")
    print()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SA Licence Disc API — Management CLI")
    sub = parser.add_subparsers(dest="command")

    # create-user
    p = sub.add_parser("create-user", help="Create a new API user")
    p.add_argument("--email", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--company", default=None)
    p.add_argument("--plan", default="free", choices=list(db.PLAN_LIMITS.keys()))

    # list-users
    sub.add_parser("list-users", help="List all users")

    # user-info
    p = sub.add_parser("user-info", help="Show detailed user info")
    p.add_argument("--email", required=True)

    # deactivate
    p = sub.add_parser("deactivate", help="Deactivate a user")
    p.add_argument("--email", required=True)

    # activate
    p = sub.add_parser("activate", help="Activate a user")
    p.add_argument("--email", required=True)

    # regen-key
    p = sub.add_parser("regen-key", help="Regenerate a user's API key")
    p.add_argument("--email", required=True)

    # transactions
    p = sub.add_parser("transactions", help="Show recent transaction logs")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--offset", type=int, default=0)

    # user-txns
    p = sub.add_parser("user-txns", help="Show transactions for a specific user")
    p.add_argument("--email", required=True)
    p.add_argument("--limit", type=int, default=20)

    # reset-usage
    sub.add_parser("reset-usage", help="Reset monthly usage counters for all users")

    # seed-demo
    sub.add_parser("seed-demo", help="Create demo users for testing")

    args = parser.parse_args()

    commands = {
        "create-user":  cmd_create_user,
        "list-users":   cmd_list_users,
        "user-info":    cmd_user_info,
        "deactivate":   cmd_deactivate,
        "activate":     cmd_activate,
        "regen-key":    cmd_regen_key,
        "transactions": cmd_transactions,
        "user-txns":    cmd_user_txns,
        "reset-usage":  cmd_reset_usage,
        "seed-demo":    cmd_seed_demo,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
