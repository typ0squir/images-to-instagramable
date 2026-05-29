# Use RunPod's official pre-cached base image with CUDA 12.4
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Force a clean BuildKit cache bust to completely bypass corrupted cloud builder cache blobs
ENV CACHE_BUST=magic_studio_v2_sdxl_2026_05_29_16_10

WORKDIR /app

# Upgrade PyTorch, torchvision, and torchaudio to a Blackwell-compatible version (Compute Capability 10.0+ / sm_100 / sm_120)
# using the official CUDA 12.4 wheel repository. This fixes the runtime health check crash loops!
RUN pip install --no-cache-dir --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Set library path to include pip-installed cuDNN 9 libraries for ONNX Runtime GPU and PyTorch compatibility
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# Set U2NET home so the downloaded model saves safely
ENV U2NET_HOME=/app/models

# Install Python dependencies, including diffusers, transformers, accelerate, peft for our AI pipeline
RUN pip install --no-cache-dir \
    runpod \
    Pillow \
    rembg \
    onnxruntime-gpu \
    numpy \
    nvidia-cudnn-cu12 \
    diffusers \
    transformers \
    accelerate \
    peft \
    opencv-python-headless

# Pre-download the rembg models during docker build
RUN python -c "import rembg; rembg.new_session('isnet-general-use')"

# Copy custom preset style images and cache script
COPY style_wood.jpg /app/style_wood.jpg
COPY style_cafe.jpg /app/style_cafe.jpg
COPY cache_models.py .

# Copy the handler code
COPY handler.py .

# Start the serverless handler
CMD ["python", "-u", "handler.py"]
