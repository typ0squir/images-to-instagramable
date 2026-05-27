# Use RunPod's official base image which has all system dependencies (libGL, libgomp, CUDA) pre-installed!
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Set U2NET home so the downloaded model saves safely in the working directory
ENV U2NET_HOME=/app/models

# We can safely use rembg[gpu] now because this base image has CUDA!
RUN pip install --no-cache-dir \
    runpod \
    Pillow \
    rembg[gpu] \
    numpy

# Copy the handler code
COPY handler.py .

# Start the serverless handler
CMD ["python", "-u", "handler.py"]
