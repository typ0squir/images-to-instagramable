FROM python:3.11

WORKDIR /app

# Ensure system dependencies for OpenCV are present (fixes libGL.so.1 crash on import)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set U2NET home so the downloaded model saves safely in the working directory
ENV U2NET_HOME=/app/models

# Install Python dependencies
RUN pip install --no-cache-dir \
    runpod \
    Pillow \
    rembg \
    numpy

# Copy the handler code
COPY handler.py .

# Start the serverless handler
CMD ["python", "-u", "handler.py"]
