#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

PIDS=()

cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    for pid in "${PIDS[@]+"${PIDS[@]}"}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    echo -e "${GREEN}All processes stopped.${NC}"
}
trap cleanup EXIT INT TERM

log() { echo -e "${CYAN}[$1]${NC} $2"; }
err() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ── Preflight checks ──

command -v python >/dev/null 2>&1 || err "python not found"
command -v temporal >/dev/null 2>&1 || err "temporal CLI not found (brew install temporal)"
command -v node >/dev/null 2>&1 || err "node not found (needed for Remotion)"

if [ ! -f "$ROOT/.env" ]; then
    err ".env file not found. Create one with at least ANTHROPIC_API_KEY=sk-..."
fi

# ── Install Python deps ──

VENV="$ROOT/.venv"
PIP="$VENV/bin/pip"
PYTHON="$VENV/bin/python"

log "SETUP" "Installing Python dependencies..."
if [ ! -f "$VENV/bin/activate" ]; then
    python -m venv "$VENV"
fi
"$PIP" install -q --upgrade pip wheel 2>&1
"$PIP" install -q -e "." 2>&1

# ── Install Remotion deps ──

REMOTION_DIR="$ROOT/video_effects/remotion"
if [ -d "$REMOTION_DIR" ] && [ ! -d "$REMOTION_DIR/node_modules" ]; then
    log "SETUP" "Installing Remotion dependencies..."
    (cd "$REMOTION_DIR" && npm install --silent)
fi

# ── Start Temporal dev server ──

start_temporal() {
    log "TEMPORAL" "Starting Temporal dev server..."
    temporal server start-dev \
        --namespace default \
        --db-filename "$ROOT/.temporal.db" \
        --log-level error \
        > /dev/null 2>&1 &
    PIDS+=($!)

    for i in {1..15}; do
        sleep 1
        temporal workflow list --namespace default > /dev/null 2>&1 && break
        if [ "$i" -eq 15 ]; then
            err "Temporal server failed to start"
        fi
    done
    log "TEMPORAL" "Server ready at localhost:7233"
}

if lsof -i :7233 >/dev/null 2>&1; then
    # Port is taken — check if the namespace is actually usable
    if temporal workflow list --namespace default > /dev/null 2>&1; then
        log "TEMPORAL" "Server already running on :7233, reusing it"
    else
        log "TEMPORAL" "Existing server on :7233 is unhealthy, killing it..."
        lsof -ti :7233 | xargs kill 2>/dev/null || true
        sleep 2
        start_temporal
    fi
else
    start_temporal
fi

# ── Start worker ──

log "WORKER" "Starting Video Effects worker..."
"$PYTHON" -m video_effects.worker > /dev/null 2>&1 &
PIDS+=($!)
sleep 1
log "WORKER" "Worker running (PID $!)"

# ── Decide interface ──

MODE="${1:-menu}"

case "$MODE" in
    web)
        log "WEB" "Starting Chainlit web interface..."
        "$VENV/bin/chainlit" run video_effects/web.py --port 8000 &
        PIDS+=($!)
        log "WEB" "Open http://localhost:8000"
        echo ""
        echo -e "${GREEN}All services running. Press Ctrl+C to stop.${NC}"
        wait
        ;;
    cli)
        shift
        log "CLI" "Running CLI workflow..."
        echo ""
        "$PYTHON" -m video_effects.cli run "$@"
        ;;
    *)
        echo ""
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}  VFX Studio — All services running${NC}"
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
        echo "  Usage:"
        echo ""
        echo "    ./run_vfx.sh web                  Start Chainlit UI at :8000"
        echo "    ./run_vfx.sh cli video.mp4 --mg   Run CLI workflow"
        echo ""
        echo "  Or run commands in another terminal:"
        echo ""
        echo "    .venv/bin/python -m video_effects.cli run video.mp4 --programmer"
        echo "    .venv/bin/chainlit run video_effects/web.py --port 8000"
        echo ""
        echo -e "  Press ${YELLOW}Ctrl+C${NC} to stop all services."
        echo ""
        wait
        ;;
esac
