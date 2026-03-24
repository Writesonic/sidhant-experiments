FROM python:3.13-slim

# System deps: ffmpeg, curl/gnupg for NodeSource, Chromium libs for Remotion, OpenCV/mediapipe libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    gnupg \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libxshmfence1 \
    fonts-liberation \
    libgl1 \
    libgles2 \
    libegl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Node.js 20 via NodeSource
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (stub package for dependency-only layer caching)
COPY pyproject.toml .
RUN mkdir -p video_effects && touch video_effects/__init__.py \
    && pip install --no-cache-dir . \
    && pip install --no-cache-dir fastapi sse-starlette uvicorn boto3 pyyaml

# Remotion npm deps
COPY video_effects/remotion/package.json video_effects/remotion/package-lock.json video_effects/remotion/
RUN cd video_effects/remotion && npm ci

# Full source
COPY . .

# Reinstall package with actual source in place
RUN pip install --no-cache-dir .

RUN mkdir -p /tmp/video_effects

ENV PYTHONUNBUFFERED=1
