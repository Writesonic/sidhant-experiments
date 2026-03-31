# Deployment Guide — Home Lab (192.168.68.60)

## Architecture

```
[Cloudflare Tunnel] --> [Ubuntu Server @ 192.168.68.60]
                            |
                            +-- docker compose
                                |-- temporal (port 7233, UI 8080)
                                |-- worker (Python + Node.js + Remotion CLI)
                                |-- api (FastAPI, port 8000)
                                |-- frontend (Next.js, port 3001)
```

**Public URL**: `https://coderunner.sidhantsriv.me`
- `/api/*` routes to API (port 8000)
- Everything else routes to frontend (port 3001)

**Local network access**:
- Frontend: `http://192.168.68.60:3001`
- API: `http://192.168.68.60:8000`
- Temporal UI: `http://192.168.68.60:8080`

## Server Details

- **Host**: `sidhant@192.168.68.60`
- **OS**: Ubuntu 24.04.4 LTS (x86_64)
- **Docker**: v29.3.0, Compose v5.1.1
- **Project path**: `~/sidhant-experiments/`
- **Cloudflare tunnel ID**: `775ef9c4-3bdb-499f-bcc5-b3e5773fccd4`
- **Tunnel config**: `~/.cloudflared/config.yml`

## Deploying Code Changes

### 1. Sync files to server

```bash
rsync -avz \
  --exclude='node_modules' \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.venv' \
  --exclude='cv_experiments/inputs' \
  --exclude='*.mp4' \
  --exclude='.next' \
  /Users/sidhant/sidhant-experiments/ \
  sidhant@192.168.68.60:~/sidhant-experiments/
```

### 2. Rebuild affected services

**Python changes** (workflow, activities, skills, api, config):
```bash
ssh sidhant@192.168.68.60 "cd ~/sidhant-experiments/video_effects && docker compose up -d --build worker api"
```

**Frontend changes** (app/src, components, pages):
```bash
ssh sidhant@192.168.68.60 "cd ~/sidhant-experiments/video_effects && docker compose build --no-cache frontend && docker compose up -d frontend"
```
`--no-cache` is required because `NEXT_PUBLIC_API_BASE` is baked in at build time via a Docker build arg. Cached layers may contain stale values.

**Remotion composition changes** (remotion/src):
```bash
# Remotion is bundled into the worker image
ssh sidhant@192.168.68.60 "cd ~/sidhant-experiments/video_effects && docker compose up -d --build worker"
```

**All services**:
```bash
ssh sidhant@192.168.68.60 "cd ~/sidhant-experiments/video_effects && docker compose down && docker compose up -d --build"
```

### 3. Verify

```bash
# Check all containers running
ssh sidhant@192.168.68.60 "cd ~/sidhant-experiments/video_effects && docker compose ps"

# Check worker loaded all activities (should be 39+)
ssh sidhant@192.168.68.60 "cd ~/sidhant-experiments/video_effects && docker compose logs worker 2>&1 | grep -E 'Activities|Failed'"

# Check for errors
ssh sidhant@192.168.68.60 "cd ~/sidhant-experiments/video_effects && docker compose logs --tail=50 worker 2>&1"
```

## Cloudflare Tunnel

### Config location

`~/.cloudflared/config.yml` on the server:

```yaml
tunnel: 775ef9c4-3bdb-499f-bcc5-b3e5773fccd4
credentials-file: /home/sidhant/.cloudflared/775ef9c4-3bdb-499f-bcc5-b3e5773fccd4.json
ingress:
  - hostname: coderunner.sidhantsriv.me
    path: /api/.*
    service: http://localhost:8000
  - hostname: coderunner.sidhantsriv.me
    service: http://localhost:3001
  - service: http_status:404
```

### Managing the tunnel

```bash
# Start (runs in foreground — use nohup or systemd for persistence)
ssh sidhant@192.168.68.60 "nohup cloudflared tunnel run coderunner > /tmp/cloudflared.log 2>&1 &"

# Check if running
ssh sidhant@192.168.68.60 "pgrep -a cloudflared"

# View logs
ssh sidhant@192.168.68.60 "tail -20 /tmp/cloudflared.log"

# Stop
ssh sidhant@192.168.68.60 "pkill cloudflared"
```

### DNS routing

The CNAME record `coderunner.sidhantsriv.me` was created via:
```bash
cloudflared tunnel route dns coderunner coderunner.sidhantsriv.me
```
This is a one-time setup — the record persists in Cloudflare DNS.

## Key Config Notes

### NEXT_PUBLIC_API_BASE

This env var is **baked into the frontend at build time** (not runtime). It's passed as a Docker build arg in `docker-compose.yml`:

```yaml
frontend:
  build:
    args:
      - NEXT_PUBLIC_API_BASE=https://coderunner.sidhantsriv.me
```

The `app/Dockerfile` receives it via `ARG`/`ENV` before `npm run build`. Changing this value requires a frontend rebuild with `--no-cache`.

### Modal auth

The worker mounts `~/.modal.toml` for Modal SDK authentication:

```yaml
volumes:
  - /home/sidhant/.modal.toml:/root/.modal.toml:ro
```

Uses absolute path (`/home/sidhant/`) not `~` — Docker compose can resolve `~` as a directory instead of a file on some systems.

If you see `"Is a directory: '/root/.modal.toml'"` in worker logs:
1. `docker compose down`
2. Verify `~/.modal.toml` is a file: `ls -la ~/.modal.toml`
3. If it's a directory, delete it and re-copy: `scp ~/.modal.toml sidhant@192.168.68.60:~/.modal.toml`
4. `docker compose up -d`

### CORS origins

The API allows requests from these origins (set in `docker-compose.yml`):

```
http://localhost:3000
http://localhost:3001
http://192.168.68.60:3001
https://coderunner.sidhantsriv.me
```

### Shared volumes

| Volume | Purpose |
|--------|---------|
| `vfx-tmp` | Uploaded videos, temp files, render output (`/tmp/video_effects`) |
| `remotion-chrome` | Cached Chromium binary for Remotion CLI |
| `generated-components` | Runtime-generated TSX components shared between worker and frontend |

## Troubleshooting

### Stale workflows in Temporal

Temporal uses in-memory storage (dev server). Stale workflows from old code versions will retry forever with missing activities. Fix: open Temporal UI (`http://192.168.68.60:8080`), find the workflow, click **Terminate**. Or restart all services (`docker compose down && docker compose up -d`) to wipe in-memory state.

### Worker missing activities

Check logs for `Failed to load skill`:
```bash
ssh sidhant@192.168.68.60 "cd ~/sidhant-experiments/video_effects && docker compose logs worker 2>&1 | grep 'Failed'"
```
Common cause: Modal auth token missing or mounted as directory (see Modal auth section above).

### Frontend shows wrong API URL

The `NEXT_PUBLIC_API_BASE` was baked at build time. Check what's in the bundle:
```bash
ssh sidhant@192.168.68.60 "cd ~/sidhant-experiments/video_effects && docker compose exec frontend grep -r 'API_BASE\|localhost:8000\|coderunner' /app/app/.next/ 2>/dev/null | head -5"
```
If wrong, rebuild with `--no-cache`.
