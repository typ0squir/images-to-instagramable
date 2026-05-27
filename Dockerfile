FROM python:3.10-slim

WORKDIR /app

# Install Python dependencies using opencv-python-headless to avoid system GUI library requirements
RUN pip install --no-cache-dir \
    runpod \
    Pillow \
    opencv-python-headless \
    rembg \
    numpy

# Pre-download rembg models (isnet-general-use) to cache them in the Docker image
RUN python -c "import rembg; rembg.new_session('isnet-general-use')"

# Copy the handler code
COPY handler.py .

# Start the serverless handler
CMD ["python", "-u", "handler.py"]
