# Use RunPod's official pre-cached base image to ensure near-zero cold starts!
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Set U2NET home so the downloaded model saves safely
ENV U2NET_HOME=/app/models

# Install Python dependencies. Since we are using virtualenv/system pip in this pre-configured image, 
# we do not need --break-system-packages.
RUN pip install --no-cache-dir \
    runpod \
    Pillow \
    "rembg[gpu]" \
    numpy

# Pre-download the rembg models (both u2net and isnet-general-use) during docker build
# so they are ready inside the container and do not need to be downloaded at runtime.
RUN python -c "import rembg; rembg.new_session('u2net'); rembg.new_session('isnet-general-use')"

# Copy the handler code
COPY handler.py .

# Start the serverless handler
CMD ["python", "-u", "handler.py"]


