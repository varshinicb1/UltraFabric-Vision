FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
# Expose PyQt requirements to headless version if running inside Docker for API
# We might need to override PyQt5 with a headless option if running without X server
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set environment variable for unbuffered output
ENV PYTHONUNBUFFERED=1

# Default command (will require X11 forwarding if running UI)
CMD ["python", "main.py"]
