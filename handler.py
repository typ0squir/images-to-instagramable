import runpod
import base64
import os
import torch
import numpy as np
from io import BytesIO
from PIL import Image, ImageOps, ImageFilter
import rembg
from diffusers import StableDiffusionInpaintPipeline

# 1. Global Model Loading & Optimization
print("Initializing RunPod Serverless Advanced AI Pipeline...")

# Initialize rembg session with GPU support
try:
    rembg_session = rembg.new_session(model_name="isnet-general-use", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print("Successfully loaded rembg GPU session (isnet-general-use).")
except Exception as e:
    print(f"Failed loading rembg GPU session: {e}. Using default session.")
    rembg_session = rembg.new_session(model_name="isnet-general-use")

# Load Stable Diffusion Inpainting pipeline
pipe = None
try:
    print("Loading RunwayML SD 1.5 Inpainting base weights...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        safety_checker=None,
        use_safetensors=True
    ).to("cuda")
    
    print("Loading Custom Instagram Aesthetic LoRA weights...")
    # Path is absolute to prevent any folder resolution issues in different RunPod paths
    lora_path = os.path.abspath("v2_aesthetic_lora_model")
    pipe.load_lora_weights(lora_path)
    
    print("Loading IP-Adapter weights for visual aesthetic guidance...")
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    
    # Apply aggressive memory optimization to support Blackwell Serverless MIG profile
    print("Applying PyTorch CPU offloading and VAE slicing optimizations...")
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    print("AI pipeline initialized successfully and ready for serverless requests!")
except Exception as e:
    print(f"WARNING: Could not load the full AI pipeline ({e}). Fast CPU/GPU fallback will be used.")

def pad_to_aspect_ratio(image, target_ratio=1.0, fill_color=(0, 0, 0, 255)):
    """Pads an image to match a specific target aspect ratio (width / height) by centering it on a new canvas."""
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
    if "A" in mode or (len(fill_color) == 4 and fill_color[3] < 255):
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
    
    # Check if the advanced AI pipeline is loaded. If not, use the fast rembg fallback.
    if not pipe:
        print("AI pipeline not active. Running fast background blur fallback...")
        mask = rembg.remove(original_img, only_mask=True, session=rembg_session)
        background_blurred = original_img.filter(ImageFilter.GaussianBlur(radius=15))
        composite = Image.composite(original_img, background_blurred, mask)
        final_img = composite.convert("RGB")
        
        # Save and return
        buffered = BytesIO()
        final_img.save(buffered, format="JPEG", quality=95)
        result_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return {
            "status": "success",
            "output_image_base64": result_base64,
            "pipeline": "fallback_blur"
        }

    try:
        # Step 1: Background Segmentation
        print("Executing zero-shot segmentation using rembg...")
        fg_rgba = rembg.remove(original_img, session=rembg_session)
        alpha = fg_rgba.split()[3]
        
        # Step 2: IP-Adapter Reference Image Preparation (Aspect Ratio Protected)
        # Using a black background as "empty space" to guide CLIP focus purely on the product
        print("Preparing aspect ratio protected IP-Adapter reference...")
        ref_padded, _, _, _, _ = pad_to_aspect_ratio(fg_rgba, target_ratio=1.0, fill_color=(0, 0, 0, 255))
        ref_rgb = ref_padded.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
        
        # Step 3: Main Generation Padding (Source Image and Masks)
        target_ratio_str = job_input.get('aspect_ratio', '1:1')
        target_ratio = 1.0 if target_ratio_str == '1:1' else 0.8
        
        print(f"Padding base image and masks to target aspect ratio ({target_ratio_str})...")
        padded_img, x_off, y_off, pad_w, pad_h = pad_to_aspect_ratio(original_img, target_ratio=target_ratio, fill_color=(0, 0, 0, 255))
        padded_alpha, _, _, _, _ = pad_to_aspect_ratio(alpha, target_ratio=target_ratio, fill_color=(0, 0, 0, 0))
        
        # Inpaint Mask: Redraw background + letterbox margins (white/255) and protect product (black/0)
        inpaint_mask = ImageOps.invert(padded_alpha)
        
        # Composite Mask: Smooth edge blending for original foreground
        composite_mask = padded_alpha.filter(ImageFilter.GaussianBlur(radius=1.5))
        
        # Resize to standard SD sizes for fast inference and zero VRAM OOMs
        sd_size = (512, 512) if target_ratio_str == '1:1' else (512, 640)
        init_sd = padded_img.resize(sd_size, Image.Resampling.LANCZOS)
        inpaint_mask_sd = inpaint_mask.resize(sd_size, Image.Resampling.NEAREST)
        composite_mask_sd = composite_mask.resize(sd_size, Image.Resampling.LANCZOS)
        
        # Step 4: AI Inpainting (Generating the 'Instagrammable' Background)
        print("Generating new background using Stable Diffusion Inpainting + LoRA + IP-Adapter...")
        prompt = job_input.get('prompt', "A smooth premium cafe table surface in sharp focus, beautifully heavy blurred window background, out of focus street, bokeh, aesthetic pinterest photography, warm natural sunlight, vivid korean cafe mood, professional interior photography, 8k resolution, photorealistic")
        negative_prompt = job_input.get('negative_prompt', "cars, traffic, messy street, clear background, sharp background, artificial, 3d render, plastic, flat lighting, harsh shadows")
        ip_adapter_scale = float(job_input.get('ip_adapter_scale', 0.65))
        
        pipe.set_ip_adapter_scale(ip_adapter_scale)
        generated_bg = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_sd,
            mask_image=inpaint_mask_sd,
            ip_adapter_image=ref_rgb,
            strength=0.95,
            guidance_scale=7.5,
            num_inference_steps=25
        ).images[0]
        
        # Step 5: Sharp Foreground Re-Composite
        print("Compositing original product foreground back over AI background...")
        hybrid_composite = Image.composite(init_sd, generated_bg, composite_mask_sd)
        
        # Step 6: Global Finishing Glaze (Harmonization)
        # Reuses the inpainting pipeline with a fully white mask to blend edges and unify lighting
        print("Applying global harmonization glaze...")
        white_mask = Image.new("L", sd_size, 255)
        denoising_strength = float(job_input.get('denoising_strength', 0.12))
        
        final_harmonized = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=hybrid_composite,
            mask_image=white_mask,
            ip_adapter_image=ref_rgb,
            strength=denoising_strength,
            guidance_scale=7.5,
            num_inference_steps=20
        ).images[0]
        
        # Step 7: Resize back to High Resolution
        # Upscale to high-resolution Instagram size (1024x1024 or 1024x1280)
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
            "pipeline": "stable_diffusion_inpainting"
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
