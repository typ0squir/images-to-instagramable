import runpod
import base64
import os
import torch
import cv2
import numpy as np
from io import BytesIO
from PIL import Image, ImageOps, ImageFilter
import rembg
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from transformers import pipeline as hf_pipeline

# 1. Global Model Loading & Optimization
print("Initializing RunPod Serverless Advanced Multi-ControlNet AI Pipeline with Pro-Cropping...")

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

# Load Stable Diffusion ControlNet Inpainting pipeline
pipe = None
try:
    print("Loading SD 1.5 ControlNet Canny and Depth models...")
    controlnet_canny = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16
    )
    controlnet_depth = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-depth",
        torch_dtype=torch.float16
    )
    
    print("Loading RunwayML SD 1.5 Inpainting base weights with Multi-ControlNet...")
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=[controlnet_depth, controlnet_canny],
        torch_dtype=torch.float16,
        safety_checker=None,
        variant="fp16",
        use_safetensors=True
    ).to("cuda")
    
    print("Loading Custom Instagram Aesthetic LoRA weights...")
    lora_path = os.path.abspath("v2_aesthetic_lora_model")
    pipe.load_lora_weights(lora_path)
    
    print("Loading IP-Adapter weights for visual aesthetic guidance...")
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    
    # Apply aggressive memory optimization to support Blackwell Serverless MIG profile
    print("Applying PyTorch CPU offloading and VAE slicing optimizations...")
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    print("Multi-ControlNet AI pipeline initialized successfully and ready!")
except Exception as e:
    print(f"WARNING: Could not load the full AI pipeline ({e}). Fast CPU/GPU fallback will be used.")

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
    mode = job_input.get('mode', 'inpaint') # Default to inpaint to fully replace messy background with premium studio space!
    
    if not image_base64:
        return {"error": "No image provided"}

    # Decode original image
    image_data = base64.b64decode(image_base64)
    original_img = Image.open(BytesIO(image_data)).convert("RGB")
    
    # Check if the advanced AI pipeline is loaded. If not, use the fast rembg fallback.
    if not pipe:
        print("AI pipeline not active. Running fast background blur fallback...")
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
        sd_size = (512, 512) if target_ratio_str == '1:1' else (512, 640)
        
        # --- PRO-LEVEL CRITICAL UPGRADE: Perform Aesthetic Crop at the absolute start of the pipeline! ---
        # This instantly changes the camera height/zoom perspective and completely eliminates black borders!
        print(f"Applying pro-level aesthetic crop to target ratio: {target_ratio_str}...")
        cropped_base = crop_to_subject_aspect_ratio(original_img, target_ratio=target_ratio)
        
        # Step 1: Background Segmentation on cropped image
        print("Executing zero-shot segmentation using rembg...")
        fg_rgba = rembg.remove(cropped_base, session=rembg_session)
        alpha = fg_rgba.split()[3]
        
        # Step 2: IP-Adapter Reference Image Preparation (Aspect Ratio Protected)
        # Using a black background as "empty space" to guide CLIP focus purely on the cropped product
        print("Preparing aspect ratio protected IP-Adapter reference...")
        ref_padded, _, _, _, _ = pad_to_aspect_ratio(fg_rgba, target_ratio=1.0, fill_color=(0, 0, 0, 255))
        ref_rgb = ref_padded.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
        
        # Setup Prompts
        prompt = job_input.get('prompt', "A smooth premium wooden cafe table surface in sharp focus, beautifully heavy blurred window background, out of focus street, bokeh, aesthetic pinterest photography, warm natural sunlight, vivid korean cafe mood, professional interior photography, 8k resolution, photorealistic")
        negative_prompt = job_input.get('negative_prompt', "cars, traffic, messy street, clear background, sharp background, artificial, 3d render, plastic, flat lighting, harsh shadows")
        ip_adapter_scale = float(job_input.get('ip_adapter_scale', 0.65))
        pipe.set_ip_adapter_scale(ip_adapter_scale)

        # Mode Routing
        if mode == "overlay":
            # --- OVERLAY MODE: 0.2 Denoising Glaze over cropped aesthetic scene ---
            print("Executing Overlay Mode (0.2 Denoising Glaze over cropped canvas)...")
            init_sd = cropped_base.resize(sd_size, Image.Resampling.LANCZOS)
            
            # Extract maps for the ControlNets even in overlay to reinforce the cropped structure
            canny_map = extract_canny_map(init_sd)
            depth_map = extract_depth_map(init_sd, sd_size)
            
            white_mask = Image.new("L", sd_size, 255)
            denoising_strength = float(job_input.get('denoising_strength', 0.20))
            
            final_harmonized = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_sd,
                mask_image=white_mask,
                control_image=[depth_map, canny_map],
                controlnet_conditioning_scale=[0.60, 0.40],
                ip_adapter_image=ref_rgb,
                strength=denoising_strength,
                guidance_scale=7.5,
                num_inference_steps=25
            ).images[0]
            
        else:
            # --- INPAINT MODE: Locked-Structure Environment Upgrade over cropped canvas ---
            print("Executing Multi-ControlNet Inpaint & Composite Mode over cropped canvas...")
            
            # Since the base image is already cropped to the target aspect ratio,
            # padding here acts as a zero-op fallback, avoiding black border creations!
            padded_img, x_off, y_off, pad_w, pad_h = pad_to_aspect_ratio(cropped_base, target_ratio=target_ratio, fill_color=(0, 0, 0, 255))
            padded_alpha, _, _, _, _ = pad_to_aspect_ratio(alpha, target_ratio=target_ratio, fill_color=(0, 0, 0, 0))
            
            inpaint_mask = ImageOps.invert(padded_alpha)
            composite_mask = padded_alpha.filter(ImageFilter.GaussianBlur(radius=1.5))
            
            # Resize
            init_sd = padded_img.resize(sd_size, Image.Resampling.LANCZOS)
            inpaint_mask_sd = inpaint_mask.resize(sd_size, Image.Resampling.NEAREST)
            composite_mask_sd = composite_mask.resize(sd_size, Image.Resampling.LANCZOS)
            
            # Extract structures using Multi-ControlNet to lock the cropped table edge and window frames
            canny_map = extract_canny_map(init_sd)
            depth_map = extract_depth_map(init_sd, sd_size)
            
            # Step 4: AI Inpainting with Multi-ControlNet locking background frames
            print("Generating new background with Multi-ControlNet constraint...")
            generated_bg = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_sd,
                mask_image=inpaint_mask_sd,
                control_image=[depth_map, canny_map],
                controlnet_conditioning_scale=[0.40, 0.20],
                ip_adapter_image=ref_rgb,
                strength=0.95,
                guidance_scale=7.5,
                num_inference_steps=25
            ).images[0]
            
            # Step 5: Sharp Foreground Re-Composite
            print("Compositing original product foreground back over AI background...")
            hybrid_composite = Image.composite(init_sd, generated_bg, composite_mask_sd)
            
            # Step 6: Global Finishing Glaze (Harmonization)
            print("Applying global harmonization glaze...")
            white_mask = Image.new("L", sd_size, 255)
            denoising_strength = float(job_input.get('denoising_strength', 0.12))
            
            final_harmonized = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=hybrid_composite,
                mask_image=white_mask,
                control_image=[depth_map, canny_map],
                controlnet_conditioning_scale=[0.45, 0.25],
                ip_adapter_image=ref_rgb,
                strength=denoising_strength,
                guidance_scale=7.5,
                num_inference_steps=20
            ).images[0]

        # Step 7: Resize back to High Resolution and Protect Brand Logos/Text
        final_output_size = (1024, 1024) if target_ratio_str == '1:1' else (1024, 1280)
        final_img = final_harmonized.resize(final_output_size, Image.Resampling.LANCZOS)
        
        # --- PRO-LEVEL CRITICAL TEXT & LOGO PROTECTION: Paste original pixels back using eroded inner mask ---
        print("Pasting razor-sharp original brand logos, text, and phone screens back onto final image...")
        try:
            high_res_original = cropped_base.resize(final_output_size, Image.Resampling.LANCZOS)
            high_res_alpha = alpha.resize(final_output_size, Image.Resampling.LANCZOS)
            
            # Erode the mask by ~15 pixels to keep border harmonization but lock inner brand marks
            high_res_inner_mask = high_res_alpha.filter(ImageFilter.MinFilter(size=15))
            # Soften the edges of the inner mask slightly for a flawless blend
            high_res_inner_mask = high_res_inner_mask.filter(ImageFilter.GaussianBlur(radius=3))
            
            # Composite the high-res original back over the AI harmonized image!
            final_img = Image.composite(high_res_original, final_img, high_res_inner_mask)
            print("Successfully protected and restored high-resolution text and brand logos!")
        except Exception as protect_err:
            print(f"Failed logo protection overlay: {protect_err}. Proceeding with base harmonized image.")
        
        # Clean up VRAM memory pointers
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("AI rendering workflow complete!")
        
        # Encode back to base64
        buffered = BytesIO()
        final_img.save(buffered, format="JPEG", quality=95)
        result_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return {
            "status": "success",
            "output_image_base64": result_base64,
            "pipeline": f"stable_diffusion_controlnet_{mode}"
        }
        
    except Exception as e:
        print(f"Error during AI rendering: {e}. Executing fast rembg blur fallback.")
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
            return {"error": f"AI pipeline failed and fallback crashed: {fallback_err}"}

# Start the serverless handler
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
