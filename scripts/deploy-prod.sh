#!/usr/bin/env bash
set -Eeuo pipefail

# CI never creates, replaces, or appends .env. Paste it on this host yourself.
if [[ ! -f .env ]]; then
  echo "Missing .env. Paste your env file onto this host at $(pwd)/.env, then redeploy." >&2
  exit 1
fi

compose=(docker compose)

"${compose[@]}" up -d --build --force-recreate

for attempt in {1..18}; do
  if curl --fail --silent --show-error http://127.0.0.1:8000/health >/dev/null; then
    "${compose[@]}" ps
    echo "Deployment succeeded."
    exit 0
  fi
  sleep 5
done

"${compose[@]}" ps >&2
"${compose[@]}" logs --tail=100 voice-agent >&2
echo "Deployment failed: the /health endpoint did not become ready." >&2
exit 1
