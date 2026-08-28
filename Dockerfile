FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer if requirements unchanged)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY config/ config/
COPY scripts/serve.py scripts/serve.py

# Copy trained model checkpoint
COPY artifacts/models/best.pt artifacts/models/best.pt

# Expose FastAPI port
EXPOSE 8000

CMD ["python3", "scripts/serve.py"]
