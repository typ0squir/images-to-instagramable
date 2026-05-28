# Use RunPod's official pre-cached base image with CUDA 12.4 to support Blackwell GPUs with near-zero cold starts!
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Upgrade PyTorch inside the pre-cached image to the latest stable release (v2.5.1+) supporting Blackwell sm_100 architecture
RUN pip install --no-cache-dir --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Set library path to include pip-installed cuDNN 9 libraries for ONNX Runtime GPU support
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# Set U2NET home so the downloaded model saves safely
ENV U2NET_HOME=/app/models


# Install Python dependencies. Since we are using virtualenv/system pip in this pre-configured image, 
# we do not need --break-system-packages.
RUN pip install --no-cache-dir \
    runpod \
    Pillow \
    "rembg[gpu]" \
    numpy \
    nvidia-cudnn-cu12

# Pre-download the rembg models (both u2net and isnet-general-use) during docker build
# so they are ready inside the container and do not need to be downloaded at runtime.
RUN python -c "import rembg; rembg.new_session('u2net'); rembg.new_session('isnet-general-use')"

# Copy the handler code
COPY handler.py .

# Start the serverless handler
CMD ["python", "-u", "handler.py"]


