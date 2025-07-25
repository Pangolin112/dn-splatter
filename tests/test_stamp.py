import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelFilter(nn.Module):
    def __init__(self, ksize=3, use_grayscale=False):
        super(SobelFilter, self).__init__()
        
        # Define Sobel kernels
        if ksize == 3:
            sobel_x = torch.tensor([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]], dtype=torch.float32)
            
            sobel_y = torch.tensor([[-1, -2, -1],
                                    [ 0,  0,  0],
                                    [ 1,  2,  1]], dtype=torch.float32)
        elif ksize == 5:
            # 5x5 Sobel kernels
            sobel_x = torch.tensor([[-1, -2, 0, 2, 1],
                                    [-4, -8, 0, 8, 4],
                                    [-6, -12, 0, 12, 6],
                                    [-4, -8, 0, 8, 4],
                                    [-1, -2, 0, 2, 1]], dtype=torch.float32)
            
            sobel_y = torch.tensor([[-1, -4, -6, -4, -1],
                                    [-2, -8, -12, -8, -2],
                                    [0, 0, 0, 0, 0],
                                    [2, 8, 12, 8, 2],
                                    [1, 4, 6, 4, 1]], dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported kernel size: {ksize}. Use 3 or 5.")
        
        # Reshape for conv2d (out_channels, in_channels, height, width)
        self.sobel_x = sobel_x.view(1, 1, ksize, ksize)
        self.sobel_y = sobel_y.view(1, 1, ksize, ksize)
        
        # Register as buffers (not trainable parameters)
        self.register_buffer('weight_x', self.sobel_x)
        self.register_buffer('weight_y', self.sobel_y)

        self.ksize = ksize
        self.use_grayscale = use_grayscale
        self.padding = ksize // 2
    
    def forward(self, x):
        """
        Apply Sobel filter to input tensor
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Edge magnitude tensor of shape (B, C, H, W)
        """
        if self.use_grayscale and x.shape[1] == 3:
            # Convert to grayscale first
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        # Ensure weights are on the same device as input
        weight_x = self.weight_x.to(x.device)
        weight_y = self.weight_y.to(x.device)

        # Handle different number of channels
        if x.shape[1] > 1:
            # Apply Sobel to each channel separately
            edges = []
            for i in range(x.shape[1]):
                channel = x[:, i:i+1, :, :]
                gx = F.conv2d(channel, weight_x, padding=self.padding)
                gy = F.conv2d(channel, weight_y, padding=self.padding)
                edge = torch.sqrt(gx**2 + gy**2 + 1e-6)  # Add small epsilon for numerical stability
                edges.append(edge)
            return torch.cat(edges, dim=1)
        else:
            # Single channel
            gx = F.conv2d(x, weight_x, padding=self.padding)
            gy = F.conv2d(x, weight_y, padding=self.padding)
            return torch.sqrt(gx**2 + gy**2 + 1e-6)


def apply_sobel_opencv_additive(stamp_img, base_img, edge_strength=0.5):
    """Apply Sobel filter using OpenCV - additive mode to preserve brightness"""
    # Convert to grayscale for edge detection
    gray_stamp = cv2.cvtColor(stamp_img, cv2.COLOR_BGR2GRAY)
    
    # Apply Sobel operator
    sobel_x = cv2.Sobel(gray_stamp, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_stamp, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate magnitude
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize edges
    magnitude = magnitude / (magnitude.max() + 1e-6) * 255 * edge_strength
    
    # Convert to 3-channel
    magnitude_3ch = cv2.cvtColor(magnitude.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    # Add edges to base image (not blend) - this preserves brightness
    rows, cols = stamp_img.shape[:2]
    result = base_img.copy()
    result[:rows, :cols] = np.clip(
        base_img[:rows, :cols].astype(np.float32) + magnitude_3ch.astype(np.float32),
        0, 255
    ).astype(np.uint8)
    
    return result


def apply_laplacian_additive(stamp_img, base_img, edge_strength=0.5):
    """Apply Laplacian filter - additive mode"""
    # Convert to grayscale
    gray_stamp = cv2.cvtColor(stamp_img, cv2.COLOR_BGR2GRAY)
    
    # Apply Laplacian
    laplacian = cv2.Laplacian(gray_stamp, cv2.CV_64F, ksize=3)
    
    # Take absolute value and normalize
    laplacian = np.abs(laplacian)
    laplacian = laplacian / (laplacian.max() + 1e-6) * 255 * edge_strength
    
    # Convert to 3-channel
    laplacian_3ch = cv2.cvtColor(laplacian.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    # Add to base image
    rows, cols = stamp_img.shape[:2]
    result = base_img.copy()
    result[:rows, :cols] = np.clip(
        base_img[:rows, :cols].astype(np.float32) + laplacian_3ch.astype(np.float32),
        0, 255
    ).astype(np.uint8)
    
    return result


def apply_canny_additive(stamp_img, base_img, low_threshold=50, high_threshold=150, edge_strength=0.5):
    """Apply Canny edge detection - additive mode"""
    # Convert to grayscale
    gray_stamp = cv2.cvtColor(stamp_img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_stamp, (5, 5), 1.4)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    # Scale edges
    edges = (edges * edge_strength).astype(np.uint8)
    
    # Convert to 3-channel
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Add to base image
    rows, cols = stamp_img.shape[:2]
    result = base_img.copy()
    result[:rows, :cols] = np.clip(
        base_img[:rows, :cols].astype(np.float32) + edges_3ch.astype(np.float32),
        0, 255
    ).astype(np.uint8)
    
    return result


def apply_emboss_filter_additive(stamp_img, base_img, strength=0.3):
    """Apply custom emboss filter - additive mode"""
    # Define emboss kernel
    emboss_kernel = np.array([[-2, -1,  0],
                              [-1,  1,  1],
                              [ 0,  1,  2]], dtype=np.float32)
    
    # Apply filter to each channel
    embossed = np.zeros_like(stamp_img, dtype=np.float32)
    for i in range(3):
        embossed[:, :, i] = cv2.filter2D(stamp_img[:, :, i], -1, emboss_kernel)
    
    # Extract only the positive edges (bright parts)
    embossed = np.maximum(embossed, 0) * strength
    
    # Add to base image
    rows, cols = stamp_img.shape[:2]
    result = base_img.copy()
    result[:rows, :cols] = np.clip(
        base_img[:rows, :cols].astype(np.float32) + embossed,
        0, 255
    ).astype(np.uint8)
    
    return result


def apply_prewitt_filter_additive(stamp_img, base_img, edge_strength=0.5):
    """Apply Prewitt edge detection filter - additive mode"""
    # Prewitt kernels
    prewitt_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]], dtype=np.float32)
    
    prewitt_y = np.array([[-1, -1, -1],
                          [ 0,  0,  0],
                          [ 1,  1,  1]], dtype=np.float32)
    
    # Convert to grayscale
    gray_stamp = cv2.cvtColor(stamp_img, cv2.COLOR_BGR2GRAY)
    
    # Apply Prewitt filters
    edge_x = cv2.filter2D(gray_stamp, cv2.CV_32F, prewitt_x)
    edge_y = cv2.filter2D(gray_stamp, cv2.CV_32F, prewitt_y)
    
    # Calculate magnitude
    magnitude = np.sqrt(edge_x**2 + edge_y**2)
    magnitude = magnitude / (magnitude.max() + 1e-6) * 255 * edge_strength
    
    # Convert to 3-channel
    magnitude_3ch = cv2.cvtColor(magnitude.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    # Add to base image
    rows, cols = stamp_img.shape[:2]
    result = base_img.copy()
    result[:rows, :cols] = np.clip(
        base_img[:rows, :cols].astype(np.float32) + magnitude_3ch.astype(np.float32),
        0, 255
    ).astype(np.uint8)
    
    return result


def apply_sobel_torch_additive(stamp_img, base_img, edge_strength=0.5):
    """Apply Sobel filter using PyTorch - additive mode"""
    # Convert images to torch tensors
    stamp_tensor = torch.from_numpy(stamp_img.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    
    # Create Sobel filter instance
    sobel = SobelFilter(ksize=3, use_grayscale=True)
    
    # Apply Sobel filter
    with torch.no_grad():
        edges = sobel(stamp_tensor)
    
    # Convert back to numpy and normalize
    edges_np = edges.squeeze(0).squeeze(0).numpy()
    edges_np = edges_np / (edges_np.max() + 1e-6) * 255 * edge_strength
    edges_np = edges_np.astype(np.uint8)
    
    # Convert to 3-channel
    edges_3ch = cv2.cvtColor(edges_np, cv2.COLOR_GRAY2BGR)
    
    # Add to base image
    rows, cols = stamp_img.shape[:2]
    result = base_img.copy()
    result[:rows, :cols] = np.clip(
        base_img[:rows, :cols].astype(np.float32) + edges_3ch.astype(np.float32),
        0, 255
    ).astype(np.uint8)
    
    return result


def apply_original_gradient(stamp_img, base_img, alpha=1.0):
    """Original horizontal gradient method"""
    rows, cols = stamp_img.shape[:2]
    result = base_img.copy()
    
    # Vectorized operation
    gradient = stamp_img[:rows, 1:cols].astype(np.float32) - stamp_img[:rows, :cols-1].astype(np.float32)
    
    # Add gradient to base image (not replace)
    result[:rows, :cols-1] = np.clip(
        base_img[:rows, :cols-1].astype(np.float32) + gradient * alpha,
        0, 255
    ).astype(np.uint8)
    
    return result


def apply_edge_enhance(stamp_img, base_img, method='sobel', edge_strength=0.5):
    """
    Apply edge enhancement while preserving original brightness
    
    Args:
        stamp_img: Reference/stamp image
        base_img: Base image to apply edges to
        method: Edge detection method ('sobel', 'laplacian', 'canny')
        edge_strength: Strength of edges (0-1)
    """
    # Convert to grayscale for edge detection
    gray_stamp = cv2.cvtColor(stamp_img, cv2.COLOR_BGR2GRAY)
    
    if method == 'sobel':
        # Sobel edge detection
        grad_x = cv2.Sobel(gray_stamp, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_stamp, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(grad_x**2 + grad_y**2)
    elif method == 'laplacian':
        edges = np.abs(cv2.Laplacian(gray_stamp, cv2.CV_64F))
    elif method == 'canny':
        edges = cv2.Canny(gray_stamp, 50, 150).astype(np.float64)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize edges to 0-1 range
    if edges.max() > 0:
        edges = edges / edges.max()
    
    # Convert edges to color (white edges)
    edges_color = np.stack([edges * 255] * 3, axis=-1) * edge_strength
    
    # Add edges to original image
    rows, cols = stamp_img.shape[:2]
    result = base_img.copy()
    result[:rows, :cols] = np.clip(
        base_img[:rows, :cols].astype(np.float32) + edges_color,
        0, 255
    ).astype(np.uint8)
    
    return result


def main():
    # Read images
    stamp_img = cv2.imread('data/ref_images/face1.jpg')
    base_img = cv2.imread('data/e9ac2fc517_original/DSC08479_original.png')
    
    if stamp_img is None or base_img is None:
        print("Error loading images")
        return

    # set edge strength
    edge_strength = 4.0
    
    # Apply different edge detection methods with brightness preservation
    results = {}
    
    # 1. Original method (horizontal gradient) - additive version
    results['Original_Additive'] = apply_original_gradient(stamp_img, base_img, alpha=edge_strength)
    
    # 2. Sobel filter (OpenCV) - additive
    results['Sobel_OpenCV_Additive'] = apply_sobel_opencv_additive(stamp_img, base_img, edge_strength=edge_strength)
    
    # 3. Laplacian filter - additive
    results['Laplacian_Additive'] = apply_laplacian_additive(stamp_img, base_img, edge_strength=edge_strength)
    
    # 4. Canny edge detection - additive
    results['Canny_Additive'] = apply_canny_additive(stamp_img, base_img, edge_strength=edge_strength)
    
    # 5. Emboss filter - additive
    results['Emboss_Additive'] = apply_emboss_filter_additive(stamp_img, base_img, strength=edge_strength)
    
    # 6. Prewitt filter - additive
    results['Prewitt_Additive'] = apply_prewitt_filter_additive(stamp_img, base_img, edge_strength=edge_strength)
    
    # 7. Sobel filter (PyTorch) - additive
    results['Sobel_PyTorch_Additive'] = apply_sobel_torch_additive(stamp_img, base_img, edge_strength=edge_strength)
    
    # 8. Edge enhance methods
    results['Edge_Enhance_Sobel'] = apply_edge_enhance(stamp_img, base_img, method='sobel', edge_strength=edge_strength)
    results['Edge_Enhance_Laplacian'] = apply_edge_enhance(stamp_img, base_img, method='laplacian', edge_strength=edge_strength)
    results['Edge_Enhance_Canny'] = apply_edge_enhance(stamp_img, base_img, method='canny', edge_strength=edge_strength)

    # Save all results
    output_dir = f'tests/output/stamp_results_bright_{edge_strength}'
    os.makedirs(output_dir, exist_ok=True)
    
    for name, result in results.items():
        output_path = os.path.join(output_dir, f"{name}.png")
        cv2.imwrite(output_path, result)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()