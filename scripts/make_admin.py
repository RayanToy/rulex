#!/usr/bin/env python
"""Назначение и снятие прав администратора.

Первый зарегистрировавшийся пользователь получает права автоматически,
остальных назначают отсюда.

    python scripts/make_admin.py --list
    python scripts/make_admin.py ivanov
    python scripts/make_admin.py ivanov --revoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.orm import Session  # noqa: E402

from database import engine  # noqa: E402
from models import User  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("username", nargs="?", help="имя пользователя")
    parser.add_argument("--revoke", action="store_true", help="снять права вместо выдачи")
    parser.add_argument("--list", action="store_true", help="показать всех пользователей")
    args = parser.parse_args()

    with Session(engine) as session:
        if args.list:
            users = session.query(User).order_by(User.id).all()
            if not users:
                print("Пользователей нет")
                return 0
            print(f"{'id':>4}  {'админ':<6} {'логин':<20} email")
            for u in users:
                print(f"{u.id:>4}  {'да' if u.is_admin else 'нет':<6} {u.username:<20} {u.email}")
            return 0

        if not args.username:
            parser.error("укажите имя пользователя или --list")

        user = session.query(User).filter(User.username == args.username).first()
        if not user:
            print(f"Пользователь '{args.username}' не найден", file=sys.stderr)
            return 1

        user.is_admin = not args.revoke
        session.commit()
        print(f"{user.username}: права администратора "
              f"{'сняты' if args.revoke else 'выданы'}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
