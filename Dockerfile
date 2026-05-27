FROM python:3.10

WORKDIR /app

# Install Python dependencies using opencv-python-headless
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
