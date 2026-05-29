import torch
from diffusers import (
    StableDiffusionXLControlNetImg2ImgPipeline,
    ControlNetModel,
    AutoencoderKL
)
from transformers import pipeline as hf_pipeline

def cache_models():
    print("Starting pre-downloading and caching of SDXL models for Blackwell GPU environments...")
    
    print("Pre-downloading and caching SDXL ControlNet Depth model...")
    controlnet_depth = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    
    print("Pre-downloading and caching SDXL ControlNet Canny model...")
    controlnet_canny = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    
    print("Pre-downloading and caching VAE fp16 fix model...")
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16
    )
    
    print("Pre-downloading and caching SDXL Base 1.0 Pipeline...")
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=[controlnet_depth, controlnet_canny],
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    
    print("Pre-downloading and caching SDXL IP-Adapter weights...")
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
    
    print("Pre-downloading and caching Intel DPT Depth Estimation model...")
    hf_pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")
    
    print("Successfully cached all SDXL Base, VAE fix, ControlNets, IP-Adapter, and Depth Estimation weights!")

if __name__ == "__main__":
    cache_models()
