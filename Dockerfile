FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps (CPU-only torch to save space)
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r backend/requirements.txt

# Copy everything
COPY . .

# Expose port
EXPOSE 5000

# Start
CMD ["python", "backend/app.py"]
