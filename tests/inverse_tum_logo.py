import cv2
import numpy as np
from PIL import Image

def convert_white_to_black_pil(input_path, output_path):
    # Open the image
    img = Image.open(input_path)
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Create a mask for white pixels
    white_mask = np.all(img_array == [255, 255, 255], axis=2)
    
    # Set white pixels to black
    img_array[white_mask] = [0, 0, 0]
    
    # Convert back to PIL Image
    result_img = Image.fromarray(img_array)
    
    # Save the image
    result_img.save(output_path)
    print(f"Image saved using PIL: {output_path}")

if __name__ == "__main__":
    input_image = "data/ref_images/tum_white.png"  # Replace with your input image path
    
    # Using PIL (recommended for most cases)
    convert_white_to_black_pil(input_image, "data/ref_images/tum_black.png")