# Use RunPod's official pre-cached base image with CUDA 12.4
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Force a clean BuildKit cache bust to completely bypass corrupted cloud builder cache blobs
ENV CACHE_BUST=magic_studio_v2_2026_05_28_17_15

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

# Copy custom LoRA weights and cache script
COPY v2_aesthetic_lora_model /app/v2_aesthetic_lora_model
COPY cache_models.py .

# Pre-download and cache Stable Diffusion, IP-Adapter, and CLIP image encoder weights are skipped to maintain a lightweight image (~1.5GB)
# RunwayML models will load dynamically and persist on RunPod's mounted network volume.

# Copy the handler code
COPY handler.py .

# Start the serverless handler
CMD ["python", "-u", "handler.py"]
