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
print("Initializing RunPod Serverless Advanced Multi-ControlNet AI Pipeline...")

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

def pad_to_aspect_ratio(image, target_ratio=1.0, fill_color=(0, 0, 0, 255)):
    """Pads an image to match a specific target aspect ratio by centering it on a new canvas."""
    w, h = image.size
    current_ratio = w / h
    
    if current_ratio > target_ratio:
        # Too wide -> Pad top and bottom
        new_w = w
        new_h = int(w / target_ratio)
    else:
        # Too tall -> Pad left and right
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
    mode = job_input.get('mode', 'inpaint') # 'inpaint' (default) or 'overlay'
    
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
        # Common Preprocessing
        target_ratio_str = job_input.get('aspect_ratio', '1:1')
        target_ratio = 1.0 if target_ratio_str == '1:1' else 0.8
        sd_size = (512, 512) if target_ratio_str == '1:1' else (512, 640)
        
        # Step 1: Background Segmentation
        print("Executing zero-shot segmentation using rembg...")
        fg_rgba = rembg.remove(original_img, session=rembg_session)
        alpha = fg_rgba.split()[3]
        
        # Step 2: IP-Adapter Reference Image Preparation (Aspect Ratio Protected)
        # Using a black background as "empty space" to guide CLIP focus purely on the product
        print("Preparing aspect ratio protected IP-Adapter reference...")
        ref_padded, _, _, _, _ = pad_to_aspect_ratio(fg_rgba, target_ratio=1.0, fill_color=(0, 0, 0, 255))
        ref_rgb = ref_padded.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
        
        # Setup Prompts
        prompt = job_input.get('prompt', "A smooth premium cafe table surface in sharp focus, beautifully heavy blurred window background, out of focus street, bokeh, aesthetic pinterest photography, warm natural sunlight, vivid korean cafe mood, professional interior photography, 8k resolution, photorealistic")
        negative_prompt = job_input.get('negative_prompt', "cars, traffic, messy street, clear background, sharp background, artificial, 3d render, plastic, flat lighting, harsh shadows")
        ip_adapter_scale = float(job_input.get('ip_adapter_scale', 0.65))
        pipe.set_ip_adapter_scale(ip_adapter_scale)

        # Mode Routing
        if mode == "overlay":
            # --- OVERLAY MODE: 0.2 Denoising Glaze over original desk & window layout ---
            print("Executing Overlay Mode (0.2 Denoising Glaze to preserve original table/window architecture)...")
            padded_img, _, _, _, _ = pad_to_aspect_ratio(original_img, target_ratio=target_ratio, fill_color=(255, 255, 255, 255))
            init_sd = padded_img.resize(sd_size, Image.Resampling.LANCZOS)
            
            # Extract maps for the ControlNets even in overlay to reinforce the structure
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
            # --- INPAINT MODE (Default): Locked-Structure Environment Upgrade via Canny + Depth ControlNet ---
            print("Executing Multi-ControlNet Inpaint & Composite Mode...")
            padded_img, x_off, y_off, pad_w, pad_h = pad_to_aspect_ratio(original_img, target_ratio=target_ratio, fill_color=(0, 0, 0, 255))
            padded_alpha, _, _, _, _ = pad_to_aspect_ratio(alpha, target_ratio=target_ratio, fill_color=(0, 0, 0, 0))
            
            # Inpaint Mask: Redraw background + letterbox margins (white/255) and protect product (black/0)
            inpaint_mask = ImageOps.invert(padded_alpha)
            
            # Composite Mask: Smooth edge blending for original foreground
            composite_mask = padded_alpha.filter(ImageFilter.GaussianBlur(radius=1.5))
            
            # Resize
            init_sd = padded_img.resize(sd_size, Image.Resampling.LANCZOS)
            inpaint_mask_sd = inpaint_mask.resize(sd_size, Image.Resampling.NEAREST)
            composite_mask_sd = composite_mask.resize(sd_size, Image.Resampling.LANCZOS)
            
            # Extract structures using Multi-ControlNet to lock the wood desk edge and window frames
            print("Extracting structural lines and depth layout to prevent distortion...")
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
                controlnet_conditioning_scale=[0.55, 0.35], # Depth holds desk outline, Canny holds detailed desk grain/straw lines
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
                controlnet_conditioning_scale=[0.60, 0.40],
                ip_adapter_image=ref_rgb,
                strength=denoising_strength,
                guidance_scale=7.5,
                num_inference_steps=20
            ).images[0]

        # Step 7: Resize back to High Resolution
        final_output_size = (1024, 1024) if target_ratio_str == '1:1' else (1024, 1280)
        final_img = final_harmonized.resize(final_output_size, Image.Resampling.LANCZOS)
        
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
        # Final emergency fallback if diffusion crashes midway
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
