FROM python:3.11-slim

WORKDIR /app

# Install system deps for torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY emb.py api.py ./

EXPOSE 47821

CMD ["uvicorn", "api:app", \
     "--host", "0.0.0.0", \
     "--port", "47821", \
     "--timeout-keep-alive", "600", \
     "--log-level", "info"]
