FROM node:20-slim

WORKDIR /app

# Install remotion deps (needed by ../remotion/src imports like @remotion/google-fonts)
COPY remotion/package.json remotion/package-lock.json ./remotion/
RUN cd remotion && npm ci

# Copy remotion source (needed for @remotion-project webpack alias in next.config.ts)
COPY remotion/src/ ./remotion/src/

# Install frontend deps
COPY app/package.json app/package-lock.json ./app/
RUN cd app && npm ci

# Copy frontend source
COPY app/ ./app/

WORKDIR /app/app

EXPOSE 3000

CMD ["npm", "run", "dev"]
