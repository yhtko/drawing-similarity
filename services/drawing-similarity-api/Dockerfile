FROM node:20-slim

WORKDIR /app
RUN apt-get update \
  && apt-get install -y --no-install-recommends poppler-utils \
  && rm -rf /var/lib/apt/lists/*

COPY package.json ./
COPY server.js ./

ENV NODE_ENV=production
CMD ["node", "server.js"]
