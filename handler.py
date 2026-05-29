import runpod
import base64
import os
import torch
import cv2
import numpy as np
from io import BytesIO
from PIL import Image, ImageOps, ImageFilter
import rembg
from diffusers import (
    StableDiffusionXLControlNetImg2ImgPipeline,
    ControlNetModel,
    AutoencoderKL
)
from transformers import pipeline as hf_pipeline

# 1. Global Model Loading & Optimization
print("Initializing RunPod Serverless Advanced SDXL Multi-ControlNet AI Pipeline...")

# Initialize rembg session with GPU support
try:
    rembg_session = rembg.new_session(model_name="isnet-general-use", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print("Successfully loaded rembg GPU session (isnet-general-use).")
except Exception as e:
    print(f"Failed loading rembg GPU session: {e}. Using default session.")
    rembg_session = rembg.new_session(model_name="isnet-general-use")

# Initialize depth estimation pipeline once globally
depth_estimator = None
try:
    print("Loading Intel DPT Depth Estimation model...")
    depth_estimator = hf_pipeline("depth-estimation", model="Intel/dpt-hybrid-midas", device=0)
    print("Successfully loaded Depth Estimation model on GPU.")
except Exception as e:
    print(f"Failed to load Depth Estimation model: {e}. Fallback depth mapping will be disabled.")

# Load Stable Diffusion XL ControlNet Img2Img pipeline
pipe = None
try:
    print("Loading SDXL ControlNet Canny and Depth models (fp16 variants)...")
    controlnet_depth = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    controlnet_canny = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    
    print("Loading stabilityai SDXL Base 1.0 with VAE fix...")
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16
    )
    
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=[controlnet_depth, controlnet_canny],
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    
    print("Loading SDXL IP-Adapter weights for aesthetic guidance...")
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
    
    # Apply aggressive memory optimization to support Blackwell Serverless MIG profile
    print("Applying PyTorch CPU offloading and VAE slicing optimizations for SDXL...")
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    print("SDXL Multi-ControlNet AI pipeline initialized successfully and ready!")
except Exception as e:
    print(f"WARNING: Could not load the full SDXL AI pipeline ({e}). Fast CPU/GPU fallback will be used.")

def extract_canny_map(img, low_threshold=100, high_threshold=200):
    """Extracts Canny Edge map as an RGB PIL Image."""
    image_np = np.array(img)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    edges = cv2.Canny(image_cv, low_threshold, high_threshold)
    edges_3c = np.stack([edges, edges, edges], axis=2)
    return Image.fromarray(edges_3c)

def extract_depth_map(img, size):
    """Extracts 3D Depth Map using the Intel DPT pipeline, upscaled back to target size."""
    if depth_estimator is None:
        # Fallback to dummy black depth map
        return Image.new("L", size, 0)
    print("Extracting 3D depth frame structure...")
    depth_result = depth_estimator(img)["depth"]
    return depth_result.resize(size, Image.Resampling.LANCZOS)

def crop_to_aspect_ratio(image, target_ratio=1.0):
    """
    Aesthetically crops an image to match a target aspect ratio (width / height) by centering the crop box.
    For tall smartphone food photos, shifts the crop window slightly upward to capture the subject and window
    perfectly while discarding boring foreground desk space.
    """
    w, h = image.size
    current_ratio = w / h
    
    if current_ratio > target_ratio:
        # Too wide -> Crop left and right sides
        new_w = int(h * target_ratio)
        new_h = h
        x1 = (w - new_w) // 2
        y1 = 0
    else:
        # Too tall -> Crop top and bottom sides
        new_w = w
        new_h = int(w / target_ratio)
        x1 = 0
        y1 = (h - new_h) // 2
        # Special upward shift (18% of cropped height) to naturally center on table-mounted food/drink subjects
        upward_shift = int((h - new_h) * 0.18)
        y1 = max(0, y1 - upward_shift)
        
    print(f"Cropped image from original {w}x{h} to proportioned {new_w}x{new_h} (Center-Top Focused)")
    return image.crop((x1, y1, x1 + new_w, y1 + new_h))

def crop_to_subject_aspect_ratio(image, target_ratio=1.0, zoom_factor=1.12):
    """
    Aesthetically crops and zooms in on the foreground subject using rembg segmentation.
    - target_ratio: 1.0 (for 1:1) or 0.8 (for 4:5)
    - zoom_factor: How tightly to focus on the subject. 1.12 means the crop box will be 
      1.12 times the size of the subject's bounding box, providing a beautiful close-up.
    """
    print("Detecting subject bounding box for pro DSLR zoom framing...")
    try:
        mask = rembg.remove(image, only_mask=True, session=rembg_session)
        bbox = mask.getbbox() # (left, upper, right, lower)
        if bbox is None:
            raise Exception("No subject detected by rembg")
    except Exception as e:
        print(f"Subject detection failed ({e}). Falling back to center-top crop.")
        return crop_to_aspect_ratio(image, target_ratio)
        
    w, h = image.size
    left, upper, right, lower = bbox
    sub_w = right - left
    sub_h = lower - upper
    
    # Calculate center of the subject
    center_x = left + sub_w / 2
    center_y = upper + sub_h / 2
    
    # Calculate target crop box size based on subject size and target aspect ratio
    if target_ratio >= 1.0:
        box_size = int(max(sub_w, sub_h) * zoom_factor)
        crop_w = box_size
        crop_h = box_size
    else:
        box_w = int(max(sub_w, sub_h * target_ratio) * zoom_factor)
        crop_w = box_w
        crop_h = int(box_w / target_ratio)

    # Limit crop size to not exceed original image dimensions
    crop_w = min(crop_w, w, h)
    crop_h = int(crop_w / target_ratio)
    if crop_h > h:
        crop_h = h
        crop_w = int(h * target_ratio)
        
    # Position the crop box centered on the subject
    x1 = int(center_x - crop_w / 2)
    y1 = int(center_y - crop_h / 2)
    
    # Special upward shift (8% of crop height) to capture a bit of background above the cups
    upward_shift = int(crop_h * 0.08)
    y1 = y1 - upward_shift
    
    # Adjust coordinates to stay within image boundaries
    x1 = max(0, min(x1, w - crop_w))
    y1 = max(0, min(y1, h - crop_h))
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    
    print(f"DSLR Zoom Crop: Focused on subject at center ({center_x:.1f}, {center_y:.1f}), cropped to {crop_w}x{crop_h}")
    return image.crop((x1, y1, x2, y2))

def pad_to_aspect_ratio(image, target_ratio=1.0, fill_color=(0, 0, 0, 255)):
    """Pads an image to match a specific target aspect ratio by centering it on a new canvas."""
    w, h = image.size
    current_ratio = w / h
    
    if current_ratio > target_ratio:
        new_w = w
        new_h = int(w / target_ratio)
    else:
        new_h = h
        new_w = int(h * target_ratio)
        
    mode = image.mode
    if mode == "L":
        color = fill_color if isinstance(fill_color, int) else fill_color[0]
        canvas = Image.new("L", (new_w, new_h), color)
    elif "A" in mode or (len(fill_color) == 4 and fill_color[3] < 255):
        canvas = Image.new("RGBA", (new_w, new_h), fill_color)
    else:
        canvas = Image.new("RGB", (new_w, new_h), fill_color[:3])
        
    x_offset = (new_w - w) // 2
    y_offset = (new_h - h) // 2
    canvas.paste(image, (x_offset, y_offset))
    return canvas, x_offset, y_offset, new_w, new_h

def handler(job):
    job_input = job['input']
    image_base64 = job_input.get('image_base64')
    
    if not image_base64:
        return {"error": "No image provided"}

    # Decode original image
    image_data = base64.b64decode(image_base64)
    original_img = Image.open(BytesIO(image_data)).convert("RGB")
    
    # Check if the advanced SDXL pipeline is loaded. If not, use the fast rembg fallback.
    if not pipe:
        print("SDXL AI pipeline not active. Running fast background blur fallback...")
        mask = rembg.remove(original_img, only_mask=True, session=rembg_session)
        background_blurred = original_img.filter(ImageFilter.GaussianBlur(radius=15))
        composite = Image.composite(original_img, background_blurred, mask)
        final_img = composite.convert("RGB")
        
        buffered = BytesIO()
        final_img.save(buffered, format="JPEG", quality=95)
        result_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return {
            "status": "success",
            "output_image_base64": result_base64,
            "pipeline": "fallback_blur"
        }

    try:
        # Core Parameters
        target_ratio_str = job_input.get('aspect_ratio', '1:1')
        target_ratio = 1.0 if target_ratio_str == '1:1' else 0.8
        
        # Native SDXL resolution planning
        if target_ratio_str == '1:1':
            sdxl_size = (1024, 1024)
        else:
            sdxl_size = (832, 1024) # Native SDXL-friendly vertical aspect ratio
            
        print(f"SDXL Target aspect ratio: {target_ratio_str}, resolution: {sdxl_size[0]}x{sdxl_size[1]}")
        
        # --- PRO-LEVEL CRITICAL UPGRADE: Perform Aesthetic Crop at the absolute start of the pipeline! ---
        # This instantly changes the camera height/zoom perspective and completely eliminates black borders!
        print(f"Applying pro-level aesthetic crop to target ratio: {target_ratio_str}...")
        cropped_base = crop_to_subject_aspect_ratio(original_img, target_ratio=target_ratio, zoom_factor=1.12)
        
        # Step 1: Background Segmentation on cropped image to lock the foreground mask
        print("Executing zero-shot segmentation using rembg...")
        fg_rgba = rembg.remove(cropped_base, session=rembg_session)
        alpha = fg_rgba.split()[3]
        
        # Step 2: IP-Adapter Reference Image Preparation (Self-contained)
        # Check if reference_base64 is provided. If not, fallback to local preset style image.
        reference_base64 = job_input.get('reference_base64')
        if reference_base64:
            print("Decoding dynamic IP-Adapter style reference from API input...")
            ref_data = base64.b64decode(reference_base64)
            ref_img = Image.open(BytesIO(ref_data)).convert("RGB")
        else:
            theme = job_input.get('theme', 'wood')
            if theme == 'cafe':
                ref_path = "style_cafe.jpg"
            else:
                ref_path = "style_wood.jpg"
                
            if os.path.exists(ref_path):
                print(f"Loading local preset reference style image: {ref_path}...")
                ref_img = Image.open(ref_path).convert("RGB")
            else:
                print("Local reference image not found. Using cropped foreground as style seed fallback.")
                ref_padded, _, _, _, _ = pad_to_aspect_ratio(fg_rgba, target_ratio=1.0, fill_color=(0, 0, 0, 255))
                ref_img = ref_padded.convert("RGB")
                
        ref_img_resized = ref_img.resize((1024, 1024), Image.Resampling.LANCZOS)
        
        # Setup Prompts
        # Detailed preset prompt designed to match raw cafe elements (iced coffee in clear plastic cup, white paper cup, phone screen)
        # but generic enough to beautifully style any wooden table composition with out-of-focus street lights.
        default_prompt = (
            "A close-up aesthetic photograph of a dark iced coffee in a clear plastic cup with a red straw, "
            "a white paper coffee cup with a black straw, and a smartphone sitting on a premium warm wooden cafe table surface. "
            "Beautifully heavy blurred window background, out of focus street, volumetric bokeh, warm natural sunlight, "
            "vivid korean cafe mood, professional interior photography, 8k resolution, photorealistic"
        )
        prompt = job_input.get('prompt', default_prompt)
        negative_prompt = job_input.get('negative_prompt', "person, hands, clear background, sharp background, artificial, 3d render, plastic, flat lighting, harsh shadows, low quality, bad aesthetics, distorted")
        
        # IP-Adapter Weight Scale tuning to prevent background overfitting while adopting premium tones
        ip_adapter_scale = float(job_input.get('ip_adapter_scale', 0.50))
        pipe.set_ip_adapter_scale(ip_adapter_scale)
        print(f"IP-Adapter Influence Scale: {ip_adapter_scale}")

        # Step 3: Extract ControlNet Maps directly at native SDXL resolution
        print("Extracting multi-structural maps at SDXL native resolution...")
        cropped_base_resized = cropped_base.resize(sdxl_size, Image.Resampling.LANCZOS)
        canny_map = extract_canny_map(cropped_base_resized)
        depth_map = extract_depth_map(cropped_base_resized, sdxl_size)
        
        # Img2Img denoising strength (Img2Img strength controls creativity vs structural rigidity)
        denoising_strength = float(job_input.get('denoising_strength', 0.45))
        print(f"SDXL Denoising Strength: {denoising_strength}")
        
        # Step 4: Run SDXL Multi-ControlNet + IP-Adapter Rendering
        print("Rendering premium aesthetic background using SDXL pipeline...")
        ai_rendered = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=cropped_base_resized,
            control_image=[depth_map, canny_map],
            ip_adapter_image=ref_img_resized,
            controlnet_conditioning_scale=[0.60, 0.40], # Depth maintains cup/straw shapes, Canny captures edges
            num_inference_steps=30,
            strength=denoising_strength,
            guidance_scale=7.0
        ).images[0]
        
        # Step 5: Sharp Foreground Re-Composite
        # Paste the original high-resolution foreground objects back over the AI-generated background
        print("Compositing original product foreground back over newly rendered background...")
        alpha_resized = alpha.resize(sdxl_size, Image.Resampling.LANCZOS)
        # Slightly blur the blending edge for a seamless lighting transition
        composite_mask = alpha_resized.filter(ImageFilter.GaussianBlur(radius=1.5))
        
        hybrid_composite = Image.composite(cropped_base_resized, ai_rendered, composite_mask)
        
        # Step 6: Brand Logos and Smartphone Screen Protection Overlay (Eroded Mask)
        # This completely guarantees that there is zero text blurring or screen content replacement!
        print("Applying eroded inner-mask overlay to protect brand logos, texts, and screens...")
        try:
            # Erode the mask by ~15 pixels to protect only the center features and keep borders harmonized
            eroded_inner_mask = alpha_resized.filter(ImageFilter.MinFilter(size=15))
            # Soften the edges of this inner shield for a flawless pixel-level blend
            eroded_inner_mask = eroded_inner_mask.filter(ImageFilter.GaussianBlur(radius=3))
            
            # Composite original sharp logo/screen pixels back onto the hybrid image
            final_img = Image.composite(cropped_base_resized, hybrid_composite, eroded_inner_mask)
            print("Successfully protected and restored high-resolution text and brand marks!")
        except Exception as protect_err:
            print(f"Failed logo protection overlay: {protect_err}. Using hybrid composite as fallback.")
            final_img = hybrid_composite
            
        # Clean up VRAM memory pointers
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("SDXL Method B pipeline rendering complete!")
        
        # Encode back to base64
        buffered = BytesIO()
        final_img.save(buffered, format="JPEG", quality=95)
        result_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return {
            "status": "success",
            "output_image_base64": result_base64,
            "pipeline": "sdxl_multi_controlnet_method_b"
        }
        
    except Exception as e:
        print(f"Error during SDXL AI rendering: {e}. Executing fast rembg blur fallback.")
        try:
            mask = rembg.remove(original_img, only_mask=True, session=rembg_session)
            background_blurred = original_img.filter(ImageFilter.GaussianBlur(radius=15))
            composite = Image.composite(original_img, background_blurred, mask)
            final_img = composite.convert("RGB")
            
            buffered = BytesIO()
            final_img.save(buffered, format="JPEG", quality=95)
            result_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return {
                "status": "success",
                "output_image_base64": result_base64,
                "pipeline": "fallback_blur_emergency",
                "error_details": str(e)
            }
        except Exception as fallback_err:
            return {"error": f"SDXL pipeline failed and fallback crashed: {fallback_err}"}

# Start the serverless handler
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
