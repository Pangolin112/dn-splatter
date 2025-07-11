import numpy as np
import cv2
from scipy.sparse import linalg as linalg
from scipy.sparse import csr_matrix
from scipy.ndimage import binary_erosion
import time
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')


def gradient_mixing_blend(source, target, mask, alpha=0.8):
    """
    Fast gradient domain mixing (approximation of Poisson)
    """
    print("Using gradient mixing...")
    
    # Ensure mask is float and normalized
    if mask.dtype == np.uint8:
        mask_float = mask.astype(np.float32) / 255.0
    else:
        mask_float = mask.astype(np.float32)
    
    # Convert mask to 3D if needed
    if mask_float.ndim == 2 and source.ndim == 3:
        mask_3d = np.stack([mask_float] * source.shape[2], axis=2)
    else:
        mask_3d = mask_float
    
    # Create smooth transition mask using multiple blur passes for better blending
    kernel_size = max(5, min(source.shape[:2]) // 50)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Apply multiple Gaussian blurs for smoother transition
    smooth_mask = mask_3d.copy()
    for _ in range(3):
        if smooth_mask.ndim == 3:
            for i in range(smooth_mask.shape[2]):
                smooth_mask[:,:,i] = cv2.GaussianBlur(smooth_mask[:,:,i], (kernel_size, kernel_size), 0)
        else:
            smooth_mask = cv2.GaussianBlur(smooth_mask, (kernel_size, kernel_size), 0)
    
    # Ensure source and target are float for blending
    source_float = source.astype(np.float32)
    target_float = target.astype(np.float32)
    
    # Blend with smooth transition
    result = smooth_mask * source_float + (1 - smooth_mask) * target_float
    
    return np.clip(result, 0, 255).astype(np.uint8)


def opencv_seamless_clone(source, target, mask):
    """
    Use OpenCV's built-in seamless cloning (fastest option)
    """
    # print("Using OpenCV seamless cloning...")
    
    try:
        # Ensure all images have the same size
        if source.shape[:2] != target.shape[:2] or source.shape[:2] != mask.shape[:2]:
            h, w = target.shape[:2]
            source = cv2.resize(source, (w, h))
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Debug mask information
        # print(f"Mask dtype: {mask.dtype}, shape: {mask.shape}")
        # print(f"Mask unique values: {np.unique(mask)}")
        # print(f"Mask white pixels: {np.sum(mask > 0)}")
        
        # Ensure mask is uint8 and binary
        if mask.dtype == bool:
            mask_uint8 = (mask * 255).astype(np.uint8)
        elif mask.dtype != np.uint8:
            mask_uint8 = (mask * 255).astype(np.uint8)
        else:
            mask_uint8 = mask.copy()
        
        # Make mask binary (0 or 255) - be more aggressive about this
        mask_uint8 = np.where(mask_uint8 > 0, 255, 0).astype(np.uint8)
        
        # print(f"Processed mask dtype: {mask_uint8.dtype}")
        # print(f"Processed mask unique values: {np.unique(mask_uint8)}")
        # print(f"Processed mask white pixels: {np.sum(mask_uint8 > 0)}")
        
        # Find contours with different retrieval modes
        contours, hierarchy = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # print(f"Found {len(contours)} contours")
        
        if len(contours) == 0:
            # Try different contour finding approach
            # print("Trying alternative contour detection...")
            contours, hierarchy = cv2.findContours(mask_uint8, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            # print(f"Alternative method found {len(contours)} contours")
        
        if len(contours) == 0:
            print("Still no contours found. Trying morphological operations...")
            # Apply morphological operations to clean up mask
            kernel = np.ones((3,3), np.uint8)
            mask_cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
            mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
            
            contours, hierarchy = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # print(f"After morphological operations: {len(contours)} contours")
            
            if len(contours) == 0:
                # print("No contours found even after cleanup. Using center of mass approach...")
                # Fallback: use center of mass
                moments = cv2.moments(mask_uint8)
                if moments['m00'] > 0:
                    center_x = int(moments['m10'] / moments['m00'])
                    center_y = int(moments['m01'] / moments['m00'])
                    center = (center_x, center_y)
                    
                    # Ensure center is within bounds and in mask region
                    h, w = mask_uint8.shape
                    center_x = max(1, min(center_x, w - 2))
                    center_y = max(1, min(center_y, h - 2))
                    
                    # Make sure center is in a white region
                    if mask_uint8[center_y, center_x] == 0:
                        # Find any white pixel
                        white_pixels = np.where(mask_uint8 > 0)
                        if len(white_pixels[0]) > 0:
                            center_y = white_pixels[0][0]
                            center_x = white_pixels[1][0]
                    
                    center = (center_x, center_y)
                    # print(f"Using center of mass: {center}")
                    
                    # Try seamless clone with center of mass
                    # result = cv2.seamlessClone(source, target, mask_uint8, center, cv2.NORMAL_CLONE)
                    result = cv2.seamlessClone(source, target, mask_uint8, center, cv2.MIXED_CLONE)
                    return result
                else:
                    raise Exception("No valid mask region found")
            else:
                mask_uint8 = mask_cleaned
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        # print(f"Largest contour area: {contour_area}")
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        # print(f"Bounding rectangle: x={x}, y={y}, w={w}, h={h}")
        
        # Ensure the bounding box is within image bounds
        img_h, img_w = target.shape[:2]
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        # Calculate center point within the bounding box
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Ensure center is within image bounds
        center_x = max(1, min(center_x, img_w - 2))
        center_y = max(1, min(center_y, img_h - 2))
        
        # Verify the center point is actually in the mask
        if mask_uint8[center_y, center_x] == 0:
            # Find nearest white pixel in the contour
            mask_points = np.column_stack(np.where(mask_uint8 > 0))
            if len(mask_points) > 0:
                # Use the first white pixel
                center_y, center_x = mask_points[0]
        
        center = (center_x, center_y)
        
        # print(f"Using center point: {center}, mask region: ({x},{y},{w},{h})")
        
        # Perform seamless cloning
        result = cv2.seamlessClone(source, target, mask_uint8, center, cv2.NORMAL_CLONE)
        
        return result
        
    except Exception as e:
        print(f"OpenCV seamless cloning failed: {e}")
        print("Falling back to gradient mixing...")
        return gradient_mixing_blend(source, target, mask)



