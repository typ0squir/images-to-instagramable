import torch
from diffusers import StableDiffusionInpaintPipeline, ControlNetModel
from transformers import pipeline as hf_pipeline

def cache_models():
    print("Starting pre-downloading and caching of RunwayML SD 1.5 Inpainting weights (fp16 variant)...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        safety_checker=None,
        variant="fp16",
        use_safetensors=True
    )
    
    print("Pre-downloading and caching IP-Adapter weights...")
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    
    print("Pre-downloading and caching SD 1.5 ControlNet Canny model...")
    ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16
    )
    
    print("Pre-downloading and caching SD 1.5 ControlNet Depth model...")
    ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-depth",
        torch_dtype=torch.float16
    )
    
    print("Pre-downloading and caching Intel DPT Depth Estimation model...")
    # Trigger download of the Intel dpt-hybrid-midas pipeline
    hf_pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")
    
    print("Successfully cached all Stable Diffusion, IP-Adapter, ControlNet, and Depth Estimation weights!")

if __name__ == "__main__":
    cache_models()
