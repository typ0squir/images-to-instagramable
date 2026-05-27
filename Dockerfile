FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

WORKDIR /app

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install critical system libraries for OpenCV, ONNXRuntime, and Python environment.
# Using --fix-missing to prevent apt-get exit code 100 from flaky mirrors.
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    python-is-python3 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set U2NET home so the downloaded model saves safely
ENV U2NET_HOME=/app/models

# Upgrade pip and install Python dependencies.
# Using --break-system-packages for Ubuntu pip safety.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --break-system-packages \
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

