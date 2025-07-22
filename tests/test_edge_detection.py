import cv2
import torchvision.transforms as transforms

# if __name__ == "__main__":
#     image_source = cv2.imread('tests/input/6/source.png')
#     image_edited = cv2.imread('tests/input/6/30000_secret_image.png')

#     img_gray = cv2.cvtColor(image_source, cv2.COLOR_BGR2GRAY)
#     # img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
#     sobelxy = cv2.Sobel(src=img_gray, ddepth = cv2.CV_64F, dx=1, dy=1, ksize=5)
#     cv2.imwrite('tests/output/6/source_sobelxy.png', sobelxy)

#     img_gray = cv2.cvtColor(image_edited, cv2.COLOR_BGR2GRAY)
#     # img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
#     sobelxy = cv2.Sobel(src=img_gray, ddepth = cv2.CV_64F, dx=1, dy=1, ksize=5)
#     cv2.imwrite('tests/output/6/edited_sobelxy.png', sobelxy)


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision


class SobelFilter(nn.Module):
    def __init__(self, ksize=3):
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
        self.padding = ksize // 2
    
    def forward(self, x):
        """
        Apply Sobel filter to input tensor
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Edge magnitude tensor of shape (B, C, H, W)
        """
        # Handle different number of channels
        if x.shape[1] > 1:
            # Apply Sobel to each channel separately
            edges = []
            for i in range(x.shape[1]):
                channel = x[:, i:i+1, :, :]
                gx = F.conv2d(channel, self.weight_x, padding=self.padding)
                gy = F.conv2d(channel, self.weight_y, padding=self.padding)
                edge = torch.sqrt(gx**2 + gy**2 + 1e-6)  # Add small epsilon for numerical stability
                edges.append(edge)
            return torch.cat(edges, dim=1)
        else:
            # Single channel
            gx = F.conv2d(x, self.weight_x, padding=self.padding)
            gy = F.conv2d(x, self.weight_y, padding=self.padding)
            return torch.sqrt(gx**2 + gy**2 + 1e-6)


class SobelEdgeLoss(nn.Module):
    def __init__(self, loss_type='l1', ksize=3):
        """
        Initialize Sobel Edge Loss
        Args:
            loss_type: 'l1', 'l2', or 'cosine' similarity
        """
        super(SobelEdgeLoss, self).__init__()
        self.sobel = SobelFilter(ksize)
        self.loss_type = loss_type
        
    def forward(self, pred, target, original_edges):
        """
        Compute edge-aware loss between predicted and target images
        Args:
            pred: Predicted image tensor (B, C, H, W)
            target: Target image tensor (B, C, H, W)
        Returns:
            Loss value
        """
        # Compute edge maps
        edges_pred = self.sobel(pred)
        edges_target = self.sobel(target)

        edges_pred_np = edges_pred.detach().cpu()
        edges_target_np = edges_target.detach().cpu()
        # Normalize each image individually
        pred_normalized = (edges_pred_np - edges_pred_np.min()) / (edges_pred_np.max() - edges_pred_np.min() + 1e-8)
        target_normalized = (edges_target_np - edges_target_np.min()) / (edges_target_np.max() - edges_target_np.min() + 1e-8)

        # Save images
        torchvision.utils.save_image(
            pred_normalized, 
            f'tests/output/6/edges_pred.png'
        )
        torchvision.utils.save_image(
            target_normalized, 
            f'tests/output/6/edges_target.png'
        )
        torchvision.utils.save_image(
            original_edges, 
            f'tests/output/6/original_edges.png'
        )
        
        # add original edge
        edges_target += original_edges

        # Compute loss
        if self.loss_type == 'l1':
            loss = F.l1_loss(edges_pred, edges_target)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(edges_pred, edges_target)
        elif self.loss_type == 'cosine':
            # Flatten and compute cosine similarity
            edges_pred_flat = edges_pred.view(edges_pred.shape[0], -1)
            edges_target_flat = edges_target.view(edges_target.shape[0], -1)
            
            # Normalize
            edges_pred_norm = F.normalize(edges_pred_flat, p=2, dim=1)
            edges_target_norm = F.normalize(edges_target_flat, p=2, dim=1)
            
            # Cosine similarity loss (1 - similarity)
            loss = 1.0 - (edges_pred_norm * edges_target_norm).sum(dim=1).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
            
        return loss


class TrainableEditedImageModel(nn.Module):
    def __init__(self, initial_image_path):
        """
        Initialize the trainable model with an edited image
        Args:
            initial_image_path: Path to the initial edited image
        """
        super(TrainableEditedImageModel, self).__init__()
        
        # Load and transform the initial image
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to [0, 1] range and CHW format
        ])
        
        image = cv2.imread(initial_image_path)
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension (1, C, H, W)
        
        # Create a trainable parameter initialized with the edited image
        self.edited_image = nn.Parameter(image_tensor, requires_grad=True)
        
    def forward(self):
        """
        Forward pass returns the trainable edited image
        Returns:
            Trainable edited image tensor (B, C, H, W)
        """
        return self.edited_image
    
    def clamp_to_valid_range(self):
        """
        Clamp the edited image to valid pixel range [0, 1]
        """
        with torch.no_grad():
            self.edited_image.clamp_(0.0, 1.0)


# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert from numpy array to PIL Image
        transforms.Resize((512, 512)),  # Resize to desired dimensions
        transforms.ToTensor(),  # Converts to [0, 1] range and CHW format
    ])
    image_source = cv2.imread('tests/input/6/face2.jpg')
    # image_source = cv2.imread('data/ref_images/dancing_lion.png')
    source = transform(image_source).unsqueeze(0)  # Add batch dimension
    # image_edited = cv2.imread('tests/input/6/30000_secret_image.png')
    # edited = transform(image_edited).unsqueeze(0)
    # edited.requires_grad = True  # Enable gradient computation for edited

    # edge_loss_fn = SobelEdgeLoss(loss_type='l1', ksize=3)
    # loss = edge_loss_fn(edited, source)
    # print(f"Sobel Edge Loss: {loss.item():.4f}")
    
    # # Compute gradients
    # loss.backward()
    # print(f"Gradient shape: {edited.grad.shape}")
    # print(f"Gradient mean: {edited.grad.mean().item():.6f}")

    # Create trainable model initialized with edited image
    model = TrainableEditedImageModel('tests/input/6/30000_secret_image.png')
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create loss function
    edge_loss_fn = SobelEdgeLoss(loss_type='l1', ksize=3)

    # add original edge
    image_original = cv2.imread('tests/input/6/target.png')
    original = transform(image_original).unsqueeze(0)  # Add batch dimension
    original_edges = SobelFilter(ksize=3)(original)
    
    # Training loop example
    num_epochs = 500
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        edited = model()
        
        # Compute loss
        loss = edge_loss_fn(edited, source, original_edges)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Clamp to valid range
        model.clamp_to_valid_range()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    # Final results
    with torch.no_grad():
        final_edited = model()
        final_loss = edge_loss_fn(final_edited, source, original_edges)
        print(f"Final Sobel Edge Loss: {final_loss.item():.4f}")
        
        # Save the final edited image
        torchvision.utils.save_image(
            final_edited, 
            'tests/output/6/final_edited_image.png'
        )