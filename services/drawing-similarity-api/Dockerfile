FROM node:20-slim

WORKDIR /app
RUN apt-get update \
  && apt-get install -y --no-install-recommends python3 python3-venv poppler-utils tesseract-ocr tesseract-ocr-jpn \
  && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV TORCH_NUM_THREADS=1
RUN python3 -m venv /opt/venv
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
  && pip install --no-cache-dir -r requirements.txt

COPY package.json ./
COPY server.js ./
COPY embed_openclip.py ./
COPY extract_shape_profile.py ./

ENV OPENCLIP_SCRIPT=/app/embed_openclip.py
ENV SHAPE_SCRIPT=/app/extract_shape_profile.py
ENV NODE_ENV=production
CMD ["node", "server.js"]
