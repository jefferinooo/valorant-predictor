FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer if requirements unchanged)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY config/ config/
COPY scripts/serve.py scripts/serve.py
COPY frontend/ frontend/

# Copy exported model weights (served with numpy; no PyTorch in the image)
COPY artifacts/models/best_weights.npz artifacts/models/best_weights.npz

# Expose FastAPI port
EXPOSE 8000

CMD ["python3", "scripts/serve.py"]
