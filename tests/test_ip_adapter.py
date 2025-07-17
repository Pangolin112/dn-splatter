import torch
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

from torchvision.utils import save_image
from PIL import Image
import numpy as np
import cv2

controlnet = ControlNetModel.from_pretrained(
  "lllyasviel/control_v11f1p_sd15_depth",
  torch_dtype=torch.float16
)

pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")
pipeline.load_ip_adapter(
  "h94/IP-Adapter",
  subfolder="models",
  weight_name="ip-adapter_sd15.bin"
)

def scannet_to_controlnet_depth(depth_path, min_depth=0.1, max_depth=10.0):
    # Load depth
    depth_mm = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    depth_m = depth_mm / 1000.0
    
    # Handle invalid depths (0 or infinity)
    depth_m[depth_mm == 0] = max_depth
    
    # Normalize and invert
    depth_clipped = np.clip(depth_m, min_depth, max_depth)
    depth_normalized = (depth_clipped - min_depth) / (max_depth - min_depth)
    depth_inverted = 1.0 - depth_normalized
    
    # Convert to 8-bit
    depth_controlnet = (depth_inverted * 255).astype(np.uint8)
    
    return depth_controlnet

depth_map_opencv = scannet_to_controlnet_depth("data/fb5a96b1a2_original/DSC02791_depth.png")

cv2.imwrite("tests/output/depth_opencv.png", depth_map_opencv)

depth_map = load_image("tests/output/depth_opencv.png")
depth_map_array = np.array(depth_map)
depth_map_image = Image.fromarray(depth_map_array)
depth_map_image.save("tests/output/depth_normalized.png")

ip_adapter_image = load_image("data/fb5a96b1a2_original/DSC02791_original.png")

# depth_map = load_image("tests/input/depth.png")
# ip_adapter_image = load_image("tests/input/statue.png")

# image_ = pipeline(
#   prompt="best quality, high quality",
#   image=depth_map,
#   ip_adapter_image=ip_adapter_image,
#   negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
# ).images[0]

image_ = pipeline(
  prompt="",
  image=depth_map,
  ip_adapter_image=ip_adapter_image,
  negative_prompt="",
).images[0]

image_.save("tests/output/DSC02791_ip_adapter.png")