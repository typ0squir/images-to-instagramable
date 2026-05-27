FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV/Pillow
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    runpod \
    Pillow \
    rembg[gpu] \
    numpy

# Pre-download rembg models (isnet-general-use) to cache them in the Docker image
RUN python -c "import rembg; rembg.new_session('isnet-general-use')"

# Copy the handler code
COPY handler.py .

# Start the serverless handler
CMD ["python", "-u", "handler.py"]
