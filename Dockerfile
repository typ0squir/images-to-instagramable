# Trigger RunPod Rebuild
FROM python:3.11

WORKDIR /app

# Set U2NET home so the downloaded model saves safely in the working directory (fixes permission crashes)
ENV U2NET_HOME=/app/models

# Install Python dependencies (opencv-python-headless is used to bypass apt-get requirements)
RUN pip install --no-cache-dir \
    runpod \
    Pillow \
    opencv-python-headless \
    rembg \
    numpy

# Copy the handler code
COPY handler.py .

# Start the serverless handler
CMD ["python", "-u", "handler.py"]
