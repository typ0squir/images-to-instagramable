import torch
from diffusers import StableDiffusionInpaintPipeline

def cache_models():
    print("Starting pre-downloading and caching of RunwayML SD 1.5 Inpainting weights...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        safety_checker=None,
        use_safetensors=True
    )
    
    print("Pre-downloading and caching IP-Adapter weights...")
    # Load IP-Adapter to trigger the download of the h94/IP-Adapter repository and CLIP Image Encoder
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    
    print("Successfully cached RunwayML Stable Diffusion Inpainting, IP-Adapter, and CLIP Image Encoder weights!")

if __name__ == "__main__":
    cache_models()
