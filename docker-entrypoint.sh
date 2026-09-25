#!/bin/sh
# Точка входа контейнера: права на каталог данных и сброс привилегий.
#
# Постоянное хранилище (Amvera, том docker compose) монтируется поверх
# /app/var и может принадлежать root — тогда пользователь app не создаст
# в нём базу, и приложение упадёт на старте. Поэтому контейнер стартует
# от root, отдаёт каталог пользователю app и запускает приложение уже
# от него. Если хранилище не даёт сменить владельца, приложение работает
# от root, и в журнал пишется предупреждение.
set -e

if [ "$(id -u)" = "0" ]; then
    mkdir -p "$RULEX_DATA_DIR"
    chown app:app "$RULEX_DATA_DIR" 2>/dev/null || true
    if setpriv --reuid=app --regid=app --init-groups test -w "$RULEX_DATA_DIR"; then
        exec setpriv --reuid=app --regid=app --init-groups "$@"
    fi
    echo "[WARN] $RULEX_DATA_DIR недоступен пользователю app на запись — работаю от root" >&2
fi

exec "$@"
