FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required by OpenCV and other libs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        libsm6 \
        libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Default environment values suitable for docker-compose
ENV PORT=8000 \
    BACKEND_URL=http://0.0.0.0:8000 \
    FRONTEND_URL=http://localhost:3000

EXPOSE 8000

# Run the Flask/Socket.IO app
CMD ["python", "app.py"]

