#!/usr/bin/env bash
# Install/refresh the org.coconutlabs.pulse-ingest launchd agent.
#
# Idempotent: safe to re-run. Ensures Postgres is up and migrations are
# applied, copies the plist into ~/Library/LaunchAgents, boots the agent out
# if it's already loaded (so a re-run picks up plist edits), then bootstraps
# it fresh into the current user's GUI domain.
#
# Usage: scripts/install-launchd.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="org.coconutlabs.pulse-ingest"
PLIST_SRC="$REPO_ROOT/ops/$LABEL.plist"
PLIST_DEST="$HOME/Library/LaunchAgents/$LABEL.plist"
UID_DOMAIN="gui/$(id -u)"

echo "==> ensuring postgresql@16 is running"
brew services start postgresql@16

echo "==> applying migrations"
( cd "$REPO_ROOT" && uv run python scripts/migrate.py )

echo "==> installing plist -> $PLIST_DEST"
mkdir -p "$HOME/Library/LaunchAgents"
cp "$PLIST_SRC" "$PLIST_DEST"

echo "==> unloading any existing $LABEL"
launchctl bootout "$UID_DOMAIN/$LABEL" 2>/dev/null || true

echo "==> bootstrapping $LABEL into $UID_DOMAIN"
launchctl bootstrap "$UID_DOMAIN" "$PLIST_DEST"

echo "==> done. verify with: launchctl list | grep pulse-ingest"
echo "    and watch:          tail -f /tmp/pulse-ingest.log"
