# # """
# # Poisson Image Editing
# # William Emmanuel
# # wemmanuel3@gatech.edu
# # CS 6745 Final Project Fall 2017
# # """

# # import numpy as np
# # from scipy.sparse import linalg as linalg
# # from scipy.sparse import lil_matrix as lil_matrix

# # # Helper enum
# # OMEGA = 0
# # DEL_OMEGA = 1
# # OUTSIDE = 2

# # # Determine if a given index is inside omega, on the boundary (del omega),
# # # or outside the omega region
# # def point_location(index, mask):
# #     if in_omega(index,mask) == False:
# #         return OUTSIDE
# #     if edge(index,mask) == True:
# #         return DEL_OMEGA
# #     return OMEGA

# # # Determine if a given index is either outside or inside omega
# # def in_omega(index, mask):
# #     return mask[index] == 1

# # # Deterimine if a given index is on del omega (boundary)
# # def edge(index, mask):
# #     if in_omega(index,mask) == False: return False
# #     for pt in get_surrounding(index):
# #         # If the point is inside omega, and a surrounding point is not,
# #         # then we must be on an edge
# #         if in_omega(pt,mask) == False: return True
# #     return False

# # # Apply the Laplacian operator at a given index
# # def lapl_at_index(source, index):
# #     i,j = index
# #     val = (4 * source[i,j])    \
# #            - (1 * source[i+1, j]) \
# #            - (1 * source[i-1, j]) \
# #            - (1 * source[i, j+1]) \
# #            - (1 * source[i, j-1])
# #     return val

# # # Find the indicies of omega, or where the mask is 1
# # def mask_indicies(mask):
# #     nonzero = np.nonzero(mask)
# #     return list(zip(nonzero[0], nonzero[1]))

# # # Get indicies above, below, to the left and right
# # def get_surrounding(index):
# #     i,j = index
# #     return [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]

# # # Create the A sparse matrix
# # def poisson_sparse_matrix(points):
# #     # N = number of points in mask
# #     N = len(points)
# #     A = lil_matrix((N,N))
# #     # Set up row for each point in mask
# #     for i,index in enumerate(points):
# #         # Should have 4's diagonal
# #         A[i,i] = 4
# #         # Get all surrounding points
# #         for x in get_surrounding(index):
# #             # If a surrounding point is in the mask, add -1 to index's
# #             # row at correct position
# #             if x not in points: continue
# #             j = points.index(x)
# #             A[i,j] = -1
# #     return A

# # # Main method
# # # Does Poisson image editing on one channel given a source, target, and mask
# # def process(source, target, mask):
# #     indicies = mask_indicies(mask)
# #     N = len(indicies)
# #     # Create poisson A matrix. Contains mostly 0's, some 4's and -1's
# #     A = poisson_sparse_matrix(indicies)
# #     # Create B matrix
# #     b = np.zeros(N)
# #     for i,index in enumerate(indicies):
# #         # Start with left hand side of discrete equation
# #         b[i] = lapl_at_index(source, index)
# #         # If on boundry, add in target intensity
# #         # Creates constraint lapl source = target at boundary
# #         if point_location(index, mask) == DEL_OMEGA:
# #             for pt in get_surrounding(index):
# #                 if in_omega(pt,mask) == False:
# #                     b[i] += target[pt]

# #     # Solve for x, unknown intensities
# #     x = linalg.cg(A, b)
# #     # Copy target photo, make sure as int
# #     composite = np.copy(target).astype(int)
# #     # Place new intensity on target at given index
# #     for i,index in enumerate(indicies):
# #         composite[index] = x[0][i]
# #     return composite

# # # Naive blend, puts the source region directly on the target.
# # # Useful for testing
# # def preview(source, target, mask):
# #     return (target * (1.0 - mask)) + (source * (mask))



# """
# Improved Poisson Image Editing
# Optimized for better performance and handling of imperfect masks
# """

# import numpy as np
# from scipy.sparse import linalg as linalg
# from scipy.sparse import csr_matrix, lil_matrix
# from scipy.ndimage import binary_erosion, binary_dilation
# import cv2

# # Helper enum
# OMEGA = 0
# DEL_OMEGA = 1
# OUTSIDE = 2

# def preprocess_mask(mask, threshold=0.5):
#     """
#     Clean up the mask to ensure it's properly binary and connected.
#     Fills small holes and removes isolated pixels.
#     """
#     # Convert to binary with threshold
#     binary_mask = (mask > threshold).astype(np.uint8)
    
#     # Fill small holes using morphological operations
#     kernel = np.ones((3,3), np.uint8)
    
#     # Close small gaps (dilate then erode)
#     closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
#     # Open to remove small isolated points (erode then dilate)
#     opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    
#     # Fill holes
#     filled = cv2.fillPoly(opened.copy(), cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], 1)
    
#     return filled

# def find_boundary_points(mask):
#     """
#     Efficiently find boundary points using morphological operations
#     """
#     # Erode the mask to find interior points
#     kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
#     interior = binary_erosion(mask, kernel)
    
#     # Boundary points are in mask but not in interior
#     boundary = mask & ~interior
    
#     return boundary

# def build_poisson_matrix_fast(mask):
#     """
#     Build the Poisson matrix more efficiently
#     """
#     h, w = mask.shape
#     # Create mapping from 2D coordinates to 1D index
#     mask_indices = np.where(mask)
#     num_pixels = len(mask_indices[0])
    
#     # Create coordinate to index mapping
#     coord_to_idx = {}
#     for idx, (i, j) in enumerate(zip(mask_indices[0], mask_indices[1])):
#         coord_to_idx[(i, j)] = idx
    
#     # Build sparse matrix
#     row_ind = []
#     col_ind = []
#     data = []
    
#     # Define neighbor offsets
#     neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
#     for idx, (i, j) in enumerate(zip(mask_indices[0], mask_indices[1])):
#         # Diagonal element
#         row_ind.append(idx)
#         col_ind.append(idx)
#         data.append(4)
        
#         # Off-diagonal elements
#         for di, dj in neighbors:
#             ni, nj = i + di, j + dj
#             if (ni, nj) in coord_to_idx:
#                 row_ind.append(idx)
#                 col_ind.append(coord_to_idx[(ni, nj)])
#                 data.append(-1)
    
#     # Create CSR matrix (more efficient for solving)
#     A = csr_matrix((data, (row_ind, col_ind)), shape=(num_pixels, num_pixels))
    
#     return A, coord_to_idx, mask_indices

# def process_fast(source, target, mask):
#     """
#     Optimized Poisson image editing
#     """
#     # Clean up the mask first
#     mask_clean = preprocess_mask(mask)
    
#     # Find boundary efficiently
#     boundary = find_boundary_points(mask_clean)
    
#     # Build matrix
#     A, coord_to_idx, mask_indices = build_poisson_matrix_fast(mask_clean)
    
#     # Build b vector
#     num_pixels = len(mask_indices[0])
#     b = np.zeros(num_pixels)
    
#     for idx, (i, j) in enumerate(zip(mask_indices[0], mask_indices[1])):
#         # Laplacian of source
#         b[idx] = 4 * source[i, j] - source[i+1, j] - source[i-1, j] - source[i, j+1] - source[i, j-1]
        
#         # Boundary conditions
#         if boundary[i, j]:
#             for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
#                 ni, nj = i + di, j + dj
#                 if not mask_clean[ni, nj]:
#                     b[idx] += target[ni, nj]
    
#     # Solve using a more efficient solver
#     x, info = linalg.cg(A, b, tol=1e-6, maxiter=2000)
    
#     if info != 0:
#         print(f"Warning: Solver did not converge (info={info})")
    
#     # Create composite
#     composite = target.copy().astype(float)
#     for idx, (i, j) in enumerate(zip(mask_indices[0], mask_indices[1])):
#         composite[i, j] = x[idx]
    
#     return np.clip(composite, 0, 255).astype(np.uint8)

# # Main processing function
# def process(source, target, mask):
#     """
#     Wrapper to use the optimized version
#     """
#     return process_fast(source, target, mask)

# # Preview function remains the same
# def preview(source, target, mask):
#     return (target * (1.0 - mask)) + (source * (mask))

# # Example usage for debugging mask issues
# def visualize_mask_issues(mask):
#     """
#     Helper function to visualize mask problems
#     """
#     import matplotlib.pyplot as plt
    
#     cleaned = preprocess_mask(mask)
#     boundary = find_boundary_points(cleaned)
    
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
#     axes[0].imshow(mask, cmap='gray')
#     axes[0].set_title('Original Mask')
    
#     axes[1].imshow(cleaned, cmap='gray')
#     axes[1].set_title('Cleaned Mask')
    
#     axes[2].imshow(boundary, cmap='gray')
#     axes[2].set_title('Boundary Points')
    
#     for ax in axes:
#         ax.axis('off')
    
#     plt.tight_layout()
#     plt.show()
    
#     print(f"Original mask points: {np.sum(mask > 0)}")
#     print(f"Cleaned mask points: {np.sum(cleaned)}")
#     print(f"Boundary points: {np.sum(boundary)}")

# """
# Poisson Image Editing
# William Emmanuel
# wemmanuel3@gatech.edu
# CS 6745 Final Project Fall 2017

# Image loading and saving adapted from blending assignment

# Usage: python main.py

# For each image set to be processed, place a folder in `input`
# Each folder should have a mask file, source file, and target file.
# Result will be saved in output directory.
# """

# import os
# import errno
# from os import path
# from glob import glob

# import cv2
# import numpy as np

# IMG_EXTENSIONS = ["png", "jpeg", "jpg", "JPG", "gif", "tiff", "tif", "raw", "bmp"]
# SRC_FOLDER = "tests/input"
# OUT_FOLDER = "tests/output"

# def collect_files(prefix, extension_list=IMG_EXTENSIONS):
#     filenames = sum(map(glob, [prefix + ext for ext in extension_list]), [])
#     return filenames

# subfolders = os.walk(SRC_FOLDER)
# next(subfolders)

# for dirpath, dirnames, fnames in subfolders:
#     image_dir = os.path.split(dirpath)[-1]
#     output_dir = os.path.join(OUT_FOLDER, image_dir)
#     print ("Processing input {i}...".format(i=image_dir))

#     # Search for images to process
#     source_names = collect_files(os.path.join(dirpath, '*source.'))
#     target_names = collect_files(os.path.join(dirpath, '*target.'))
#     mask_names = collect_files(os.path.join(dirpath, '*mask.'))

#     if not len(source_names) == len(target_names) == len(mask_names) == 1:
#         print("There must be one source, one target, and one mask per input.")
#         continue

#     # Read images
#     source_img = cv2.imread(source_names[0], cv2.IMREAD_COLOR)
#     target_img = cv2.imread(target_names[0], cv2.IMREAD_COLOR)
#     mask_img = cv2.imread(mask_names[0], cv2.IMREAD_GRAYSCALE)

#     # Normalize mask to range [0,1]
#     mask = np.atleast_3d(mask_img).astype(float) / 255.
#     # Make mask binary
#     mask[mask != 1] = 0
#     # Trim to one channel
#     mask = mask[:,:,0]
#     channels = source_img.shape[-1]
#     # Call the poisson method on each individual channel
#     result_stack = [process(source_img[:,:,i], target_img[:,:,i], mask) for i in range(channels)]
#     # Merge the channels back into one image
#     result = cv2.merge(result_stack)
#     # Make result directory if needed
#     try:
#         os.makedirs(output_dir)
#     except OSError as exception:
#         if exception.errno != errno.EEXIST:
#             raise
#     # Write result
#     result = np.clip(result, 0, 255).astype(np.uint8)
#     cv2.imwrite(path.join(output_dir, 'result.png'), result)
#     print ("Finished processing input {i}.".format(i=image_dir))

import numpy as np
import cv2
from scipy.sparse import linalg as linalg
from scipy.sparse import csr_matrix
from scipy.ndimage import binary_erosion
import time
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')

# Fast implementation with multiple optimization strategies

def fast_poisson_blend(source, target, mask, method='pyramid', max_resolution=512):
    """
    Fast Poisson blending with multiple optimization strategies
    
    Args:
        source: Source image
        target: Target image  
        mask: Binary mask
        method: 'pyramid', 'downsample', 'opencv', or 'gradient_mix'
        max_resolution: Maximum resolution for processing
    """
    
    if method == 'opencv':
        return opencv_seamless_clone(source, target, mask)
    elif method == 'pyramid':
        return pyramid_poisson_blend(source, target, mask, max_resolution)
    elif method == 'downsample':
        return downsample_poisson_blend(source, target, mask, max_resolution)
    elif method == 'gradient_mix':
        return gradient_mixing_blend(source, target, mask)
    else:
        return optimized_poisson_process(source, target, mask)

def opencv_seamless_clone(source, target, mask):
    """
    Use OpenCV's built-in seamless cloning (fastest option)
    """
    print("Using OpenCV seamless cloning...")
    
    try:
        # Ensure all images have the same size
        if source.shape[:2] != target.shape[:2] or source.shape[:2] != mask.shape[:2]:
            h, w = target.shape[:2]
            source = cv2.resize(source, (w, h))
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Debug mask information
        print(f"Mask dtype: {mask.dtype}, shape: {mask.shape}")
        print(f"Mask unique values: {np.unique(mask)}")
        print(f"Mask white pixels: {np.sum(mask > 0)}")
        
        # Ensure mask is uint8 and binary
        if mask.dtype == bool:
            mask_uint8 = (mask * 255).astype(np.uint8)
        elif mask.dtype != np.uint8:
            mask_uint8 = (mask * 255).astype(np.uint8)
        else:
            mask_uint8 = mask.copy()
        
        # Make mask binary (0 or 255) - be more aggressive about this
        mask_uint8 = np.where(mask_uint8 > 0, 255, 0).astype(np.uint8)
        
        print(f"Processed mask dtype: {mask_uint8.dtype}")
        print(f"Processed mask unique values: {np.unique(mask_uint8)}")
        print(f"Processed mask white pixels: {np.sum(mask_uint8 > 0)}")
        
        # Save debug mask to check what we're working with
        cv2.imwrite("tests/output/2/debug_mask.png", mask_uint8)
        print("Saved debug mask to tests/output/2/debug_mask.png")
        
        # Find contours with different retrieval modes
        contours, hierarchy = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"Found {len(contours)} contours")
        
        if len(contours) == 0:
            # Try different contour finding approach
            print("Trying alternative contour detection...")
            contours, hierarchy = cv2.findContours(mask_uint8, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            print(f"Alternative method found {len(contours)} contours")
        
        if len(contours) == 0:
            print("Still no contours found. Trying morphological operations...")
            # Apply morphological operations to clean up mask
            kernel = np.ones((3,3), np.uint8)
            mask_cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
            mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
            
            contours, hierarchy = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"After morphological operations: {len(contours)} contours")
            
            if len(contours) == 0:
                print("No contours found even after cleanup. Using center of mass approach...")
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
                    print(f"Using center of mass: {center}")
                    
                    # Try seamless clone with center of mass
                    result = cv2.seamlessClone(source, target, mask_uint8, center, cv2.NORMAL_CLONE)
                    return result
                else:
                    raise Exception("No valid mask region found")
            else:
                mask_uint8 = mask_cleaned
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        print(f"Largest contour area: {contour_area}")
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        print(f"Bounding rectangle: x={x}, y={y}, w={w}, h={h}")
        
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
        
        print(f"Using center point: {center}, mask region: ({x},{y},{w},{h})")
        
        # Perform seamless cloning
        result = cv2.seamlessClone(source, target, mask_uint8, center, cv2.NORMAL_CLONE)
        
        return result
        
    except Exception as e:
        print(f"OpenCV seamless cloning failed: {e}")
        print("Falling back to gradient mixing...")
        return gradient_mixing_blend(source, target, mask)

def pyramid_poisson_blend(source, target, mask, max_resolution=512):
    """
    Multi-resolution pyramid approach for fast processing
    """
    print("Using pyramid blending...")
    
    h, w = source.shape[:2]
    if max(h, w) <= max_resolution:
        return optimized_poisson_process(source, target, mask)
    
    # Calculate downscale factor
    scale_factor = max_resolution / max(h, w)
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    
    # Downsample all images
    source_small = cv2.resize(source, (new_w, new_h), interpolation=cv2.INTER_AREA)
    target_small = cv2.resize(target, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask_small = cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    mask_small = (mask_small > 0).astype(np.uint8)
    
    # Process at low resolution
    if source_small.ndim == 3:
        result_small_stack = []
        for i in range(source_small.shape[2]):
            result_channel = optimized_poisson_process(
                source_small[:,:,i], target_small[:,:,i], mask_small
            )
            result_small_stack.append(result_channel)
        result_small = np.stack(result_small_stack, axis=2)
    else:
        result_small = optimized_poisson_process(source_small, target_small, mask_small)
    
    # Upsample result
    result = cv2.resize(result_small, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Refine boundaries at full resolution
    result = refine_boundaries(source, target, mask, result)
    
    return result

def downsample_poisson_blend(source, target, mask, max_resolution=512):
    """
    Simple downsample approach
    """
    print("Using downsample blending...")
    
    h, w = source.shape[:2]
    if max(h, w) <= max_resolution:
        return optimized_poisson_process(source, target, mask)
    
    # Calculate downscale factor
    scale_factor = max_resolution / max(h, w)
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    
    # Downsample
    source_small = cv2.resize(source, (new_w, new_h))
    target_small = cv2.resize(target, (new_w, new_h))
    mask_small = cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    mask_small = (mask_small > 0).astype(np.uint8)
    
    # Process
    if source_small.ndim == 3:
        result_small_stack = []
        for i in range(source_small.shape[2]):
            result_channel = optimized_poisson_process(
                source_small[:,:,i], target_small[:,:,i], mask_small
            )
            result_small_stack.append(result_channel)
        result_small = np.stack(result_small_stack, axis=2)
    else:
        result_small = optimized_poisson_process(source_small, target_small, mask_small)
    
    # Upsample
    result = cv2.resize(result_small, (w, h))
    
    return result

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

def optimized_poisson_process(source, target, mask):
    """
    Optimized Poisson processing with sparse matrix optimizations
    """
    # Erode mask slightly to avoid boundary issues
    eroded_mask = binary_erosion(mask, structure=np.ones((3,3)))
    
    # Get mask indices more efficiently
    mask_coords = np.column_stack(np.where(eroded_mask))
    n_pixels = len(mask_coords)
    
    if n_pixels == 0:
        return target
    
    print(f"Processing {n_pixels} pixels...")
    
    # Create coordinate to index mapping
    coord_to_idx = {tuple(coord): i for i, coord in enumerate(mask_coords)}
    
    # Build sparse matrix more efficiently
    row_indices = []
    col_indices = []
    data = []
    
    b = np.zeros(n_pixels)
    
    for i, (y, x) in enumerate(mask_coords):
        # Diagonal element
        row_indices.append(i)
        col_indices.append(i)
        data.append(4.0)
        
        # Laplacian of source
        b[i] = (4 * source[y, x] - 
                source[y-1, x] - source[y+1, x] - 
                source[y, x-1] - source[y, x+1])
        
        # Neighboring pixels
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            
            if (ny, nx) in coord_to_idx:
                # Interior pixel
                j = coord_to_idx[(ny, nx)]
                row_indices.append(i)
                col_indices.append(j)
                data.append(-1.0)
            else:
                # Boundary pixel
                b[i] += target[ny, nx]
    
    # Create sparse matrix
    A = csr_matrix((data, (row_indices, col_indices)), shape=(n_pixels, n_pixels))
    
    # Solve system
    x, info = linalg.cg(A, b, maxiter=min(1000, n_pixels))
    
    # Reconstruct image
    result = target.copy().astype(np.float64)
    for i, (y, x) in enumerate(mask_coords):
        result[y, x] = x[i]
    
    return np.clip(result, 0, 255).astype(np.uint8)

def refine_boundaries(source, target, mask, coarse_result, width=10):
    """
    Refine boundaries at full resolution
    """
    # Create boundary mask
    kernel = np.ones((3, 3), np.uint8)
    boundary = cv2.dilate(mask.astype(np.uint8), kernel, iterations=width) - \
               cv2.erode(mask.astype(np.uint8), kernel, iterations=width)
    
    # Apply gradient blending only at boundaries
    if boundary.sum() > 0:
        if coarse_result.ndim == 3:
            for i in range(coarse_result.shape[2]):
                boundary_region = optimized_poisson_process(
                    source[:,:,i], coarse_result[:,:,i], boundary
                )
                coarse_result[:,:,i] = np.where(boundary, boundary_region, coarse_result[:,:,i])
        else:
            boundary_region = optimized_poisson_process(source, coarse_result, boundary)
            coarse_result = np.where(boundary, boundary_region, coarse_result)
    
    return coarse_result

def process_channel_parallel(args):
    """Helper function for parallel processing"""
    source_channel, target_channel, mask, method = args
    return fast_poisson_blend(source_channel, target_channel, mask, method)

def fast_process_all_channels(source, target, mask, method='opencv', use_parallel=True):
    """
    Process all channels with optional parallelization
    """
    if source.ndim == 2:
        return fast_poisson_blend(source, target, mask, method)
    
    channels = source.shape[2]
    
    if use_parallel and channels > 1:
        # Parallel processing
        args_list = [(source[:,:,i], target[:,:,i], mask, method) for i in range(channels)]
        with Pool(processes=min(channels, 4)) as pool:
            result_channels = pool.map(process_channel_parallel, args_list)
        
        result = np.stack(result_channels, axis=2)
    else:
        # Sequential processing
        result_channels = []
        for i in range(channels):
            channel_result = fast_poisson_blend(source[:,:,i], target[:,:,i], mask, method)
            result_channels.append(channel_result)
        
        result = np.stack(result_channels, axis=2)
    
    return result

# Updated main processing function
def fast_poisson_main():
    """
    Updated main function with timing and method comparison
    """
    import os
    from glob import glob
    
    SRC_FOLDER = "tests/input"
    OUT_FOLDER = "tests/output"
    
    subfolders = os.walk(SRC_FOLDER)
    next(subfolders)
    
    methods = ['opencv', 'pyramid', 'downsample', 'gradient_mix']
    
    for dirpath, dirnames, fnames in subfolders:
        image_dir = os.path.split(dirpath)[-1]
        print(f"\nProcessing input {image_dir}...")
        
        # Load images
        source_names = glob(os.path.join(dirpath, '*source.*'))
        target_names = glob(os.path.join(dirpath, '*target.*'))
        mask_names = glob(os.path.join(dirpath, '*mask.*'))
        
        if not len(source_names) == len(target_names) == len(mask_names) == 1:
            print("Missing required files")
            continue
        
        source_img = cv2.imread(source_names[0], cv2.IMREAD_COLOR)
        target_img = cv2.imread(target_names[0], cv2.IMREAD_COLOR)
        mask_img = cv2.imread(mask_names[0], cv2.IMREAD_GRAYSCALE)
        
        print(f"Image size: {source_img.shape}")
        
        # Normalize mask
        mask = (mask_img > 127).astype(np.uint8)
        
        # Test different methods
        for method in methods:
            start_time = time.time()
            
            try:
                result = fast_process_all_channels(source_img, target_img, mask, method)
                
                elapsed = time.time() - start_time
                print(f"{method}: {elapsed:.2f}s")
                
                # Save result
                output_dir = os.path.join(OUT_FOLDER, image_dir)
                os.makedirs(output_dir, exist_ok=True)
                cv2.imwrite(os.path.join(output_dir, f'result_{method}.png'), result)
                
            except Exception as e:
                print(f"{method} failed: {e}")

# Simple usage example
def quick_blend_example():
    """
    Quick example of fast blending with error handling
    """

    input_dir_name = "2"

    try:
        # Load your images
        source = cv2.imread(f"tests/input/{input_dir_name}/30000_secret_image.png")
        target = cv2.imread(f"tests/input/{input_dir_name}/target.png") 
        mask = cv2.imread(f"tests/input/{input_dir_name}/mask.png", cv2.IMREAD_GRAYSCALE)
        
        if source is None or target is None or mask is None:
            print("Error: Could not load images. Check file paths.")
            return None
        
        # Ensure all images are the same size
        h, w = target.shape[:2]
        source = cv2.resize(source, (w, h))
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Normalize mask
        mask = (mask > 127).astype(np.uint8)
        
        print(f"Image size: {source.shape}")
        print(f"Mask has {np.sum(mask > 0)} white pixels")
        
        # Create output directory
        import os
        os.makedirs(f"tests/output/{input_dir_name}", exist_ok=True)
        
        # Try different methods in order of preference
        methods_to_try = [
            ('opencv', opencv_seamless_clone),
            ('gradient_mix', lambda s, t, m: gradient_mixing_blend(s, t, m)),
            ('pyramid', lambda s, t, m: pyramid_poisson_blend(s, t, m, 512))
        ]
        
        for method_name, method_func in methods_to_try:
            try:
                print(f"\nTrying {method_name} method...")
                start_time = time.time()
                
                print(source, target, mask)

                result = method_func(source, target, mask)
                
                elapsed = time.time() - start_time
                print(f"{method_name} method: {elapsed:.2f}s")
                
                # Save result
                output_path = f"tests/output/{input_dir_name}/result_{method_name}.png"
                cv2.imwrite(output_path, result)
                print(f"Saved result to: {output_path}")
                
                return result
                
            except Exception as e:
                print(f"{method_name} method failed: {e}")
                continue
        
        print("All methods failed!")
        return None
        
    except Exception as e:
        print(f"Error in quick_blend_example: {e}")
        return None

if __name__ == "__main__":
    # Run the fast processing
    quick_blend_example()