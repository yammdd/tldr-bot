#!/bin/sh
set -eu

DATA_DIR="${DATA_DIR:-/app/data}"
mkdir -p "$DATA_DIR"

# Default healthcheck port inside containers when neither HEALTH_PORT nor PaaS PORT is set.
if [ -z "${HEALTH_PORT:-}" ] && [ -z "${PORT:-}" ]; then
    export HEALTH_PORT=8080
fi

# Railway and cloud hosts mount fresh volumes as root; fix ownership then drop privileges.
if [ "$(id -u)" = "0" ]; then
    chown -R 10001:10001 "$DATA_DIR"
    exec setpriv --reuid=10001 --regid=10001 --init-groups "$@"
fi

exec "$@"
