#!/usr/bin/env bash
# Install/refresh the pulse launchd agents:
#   org.coconutlabs.pulse-ingest      -- the poller, StartInterval 60s
#   org.coconutlabs.pulse-caffeinate  -- caffeinate -s -i, KeepAlive, so the
#                                        Mac doesn't sleep out from under
#                                        ingestion while on AC power
#   org.coconutlabs.pulse-publish     -- pushes a status snapshot + the
#                                        derived gap ledger to Cloudflare D1
#                                        (pulse-status) every 300s. This is
#                                        the ONLY thing that talks to the
#                                        public surface -- it pushes out,
#                                        never serves in. See
#                                        scripts/publish-status.py.
#
# Idempotent: safe to re-run. Ensures Postgres is up and migrations are
# applied, then installs each plist into ~/Library/LaunchAgents, boots it
# out if already loaded (so a re-run picks up plist edits), then bootstraps
# it fresh into the current user's GUI domain. Ingest is installed first and
# fully re-bootstrapped before caffeinate/publish are touched, so a failure
# in a later step can't leave ingest unloaded.
#
# Usage: scripts/install-launchd.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UID_DOMAIN="gui/$(id -u)"

install_agent() {
    local label="$1"
    local plist_src="$REPO_ROOT/ops/$label.plist"
    local plist_dest="$HOME/Library/LaunchAgents/$label.plist"

    echo "==> installing plist -> $plist_dest"
    cp "$plist_src" "$plist_dest"

    echo "==> unloading any existing $label"
    launchctl bootout "$UID_DOMAIN/$label" 2>/dev/null || true

    echo "==> bootstrapping $label into $UID_DOMAIN"
    launchctl bootstrap "$UID_DOMAIN" "$plist_dest"
}

echo "==> ensuring postgresql@16 is running"
brew services start postgresql@16

echo "==> applying migrations"
( cd "$REPO_ROOT" && uv run python scripts/migrate.py )

mkdir -p "$HOME/Library/LaunchAgents"

install_agent org.coconutlabs.pulse-ingest
install_agent org.coconutlabs.pulse-caffeinate
install_agent org.coconutlabs.pulse-publish

echo "==> done. verify with: launchctl list | grep pulse"
echo "    ingest logs:           tail -f /tmp/pulse-ingest.log"
echo "    publish logs:          tail -f /tmp/pulse-publish.log"
echo "    caffeinate assertion:  pmset -g assertions | grep -i caffeinate"
echo "    (caffeinate -s only holds sleep at bay on AC power -- on battery"
echo "     the machine can still sleep; poll_runs records that gap honestly)"
