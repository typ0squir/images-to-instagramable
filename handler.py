import runpod
import base64
from io import BytesIO
from PIL import Image, ImageFilter
import rembg

# Initialize session with GPU support (falls back to CPU if not available)
try:
    session = rembg.new_session(model_name="isnet-general-use", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print("Successfully initialized rembg session with GPU (isnet-general-use).")
except Exception as e:
    print(f"Failed to initialize GPU session: {e}. Falling back to default session.")
    session = rembg.new_session(model_name="isnet-general-use")

def handler(job):
    job_input = job['input']
    image_base64 = job_input.get('image_base64')
    
    if not image_base64:
        return {"error": "No image provided"}

    # Decode image
    image_data = base64.b64decode(image_base64)
    original_img = Image.open(BytesIO(image_data)).convert("RGBA")

    # 1. 누끼 따기 (isnet-general-use)
    # Note: In a real production deployment, the model would be pre-downloaded in the Dockerfile
    mask = rembg.remove(original_img, only_mask=True, session=session)
    
    # 2. 배경 블러 처리 (심도 모방)
    background_blurred = original_img.filter(ImageFilter.GaussianBlur(radius=15))
    
    # 3. 원본 전경(글씨 보존) 덮어씌우기
    composite = Image.composite(original_img, background_blurred, mask)
    final_img = composite.convert("RGB")

    # Encode back to base64
    buffered = BytesIO()
    final_img.save(buffered, format="JPEG", quality=95)
    result_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {
        "status": "success",
        "output_image_base64": result_base64
    }

runpod.serverless.start({"handler": handler})
