import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import cv2
import os

np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

def save_images_for_poisson_editing(img, image_original_resized, mask, output_folder="tests/input/2"):
    """
    Save images in the format expected by Poisson image editing:
    - source: the reference image (img)
    - target: the original image (image_original_resized)
    - mask: the segmentation mask
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Convert PIL images to numpy arrays if needed
    if isinstance(img, Image.Image):
        source_array = np.array(img)
    else:
        source_array = img
    
    if isinstance(image_original_resized, Image.Image):
        target_array = np.array(image_original_resized)
    else:
        target_array = image_original_resized
    
    # Ensure mask is in the right format (0-255, uint8)
    if mask.dtype == bool:
        mask_array = (mask * 255).astype(np.uint8)
    else:
        mask_array = (mask * 255).astype(np.uint8)
    
    # Save source image (reference image with face/object)
    source_path = os.path.join(output_folder, "source.png")
    if isinstance(img, Image.Image):
        img.save(source_path)
    else:
        cv2.imwrite(source_path, cv2.cvtColor(source_array, cv2.COLOR_RGB2BGR))
    print(f"Saved source image to: {source_path}")
    
    # Save target image (original background)
    target_path = os.path.join(output_folder, "target.png")
    if isinstance(image_original_resized, Image.Image):
        image_original_resized.save(target_path)
    else:
        cv2.imwrite(target_path, cv2.cvtColor(target_array, cv2.COLOR_RGB2BGR))
    print(f"Saved target image to: {target_path}")
    
    # Save mask image
    mask_path = os.path.join(output_folder, "mask.png")
    cv2.imwrite(mask_path, mask_array)
    print(f"Saved mask image to: {mask_path}")
    
    # Optional: Save a preview of what will be blended
    preview_path = os.path.join(output_folder, "preview.png")
    mask_3d = np.stack([mask] * 3, axis=-1)
    preview = np.where(mask_3d, source_array, target_array)
    cv2.imwrite(preview_path, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    print(f"Saved preview image to: {preview_path}")
    
    print(f"\nImages saved to {output_folder}/")
    print("Files created:")
    print("- source.png (reference image)")
    print("- target.png (original background)")
    print("- mask.png (segmentation mask)")
    print("- preview.png (simple blend preview)")
    
    return source_path, target_path, mask_path, preview_path

# Load and preprocess the image
render_size = 512
contrast = 4.0
add_noise = False
noise_value = 0.05
# ref_name = "tum_white.png"
# ref_name = "face1.jpg"
# ref_name = "face2.jpg"
# ref_name = "yellow_dog.jpg"
ref_name = "dancing_lion.png"

img = Image.open(f"data/ref_images/{ref_name}").convert('RGB').resize((render_size, render_size))
img = torchvision.transforms.ColorJitter(contrast=(contrast, contrast))(img)

# set predictor
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large")
# set image
predictor.set_image(img)

input_point = np.array([[330, 256], [150, 256]])
input_label = np.array([1, 1])

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
sorted_ind = np.argsort(scores)[::-1]
masks = masks[sorted_ind]
scores = scores[sorted_ind]
logits = logits[sorted_ind]

# show_masks(img, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)

# image_original = Image.open("data/fb5a96b1a2_original/DSC02791_original.png")
# image_original = Image.open("data/49a82360aa_original/DSC00043_original.png")
image_original = Image.open("data/0cf2e9402d_original/DSC00356_original.png")

# select the mask that has the most white pixels (highest coverage area)
# Calculate the number of white pixels (True values) in each mask
mask_pixel_counts = [np.sum(mask) for mask in masks]

# Find the index of the mask with the most white pixels
max_pixels_idx = np.argmax(mask_pixel_counts)

# Select the mask with the most white pixels
if ref_name == "tum_white.png":
    mask = masks[0]
else:
    mask = masks[max_pixels_idx]

# First, ensure both images have the same size
# Resize image_original to match the size of img (render_size x render_size)
image_original_resized = image_original.resize((render_size, render_size))

# Convert images to numpy arrays
img_array = np.array(img)
image_original_array = np.array(image_original_resized)

# The mask from SAM2 is a 2D array, we need to expand it to 3D for RGB channels
# mask shape is (H, W), we need (H, W, 3)
mask_3d = np.stack([mask] * 3, axis=-1)

# Create the composite image
# Where mask is True (1), use img; where mask is False (0), use image_original
composite_image = np.where(mask_3d, img_array, image_original_array)

# Convert back to PIL Image
result_image = Image.fromarray(composite_image.astype(np.uint8))

# Display the result
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(img)
plt.title('Image with Face (img)')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(image_original_resized)
plt.title('Original Background')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(mask, cmap='gray')
plt.title('Mask')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(result_image)
plt.title('Composite Result')
plt.axis('off')

plt.tight_layout()
plt.show()

# Save images for Poisson image editing
print("\n" + "="*50)
print("SAVING IMAGES FOR POISSON IMAGE EDITING")
print("="*50)

input_dir_name = "7"

source_path, target_path, mask_path, preview_path = save_images_for_poisson_editing(
    img=img,  # Reference image (source)
    image_original_resized=image_original_resized,  # Original background (target)
    mask=mask,  # Segmentation mask
    output_folder=f"tests/output/{input_dir_name}"
)

print(f"\nYou can now run Poisson image editing with:")
print(f"python tests/test_pie.py")
print(f"\nThe result will be saved to: tests/output/{input_dir_name}/result.png")