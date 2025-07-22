import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

class StampEffect:
    def __init__(self):
        pass
    
    def create_rubber_stamp(self, text, size=(200, 200), 
                           border_thickness=10, rotation=15):
        """
        Create a rubber stamp effect with text
        """
        # Create a blank image
        img = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Try to use a bold font, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Get text dimensions
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Calculate position to center text
        x = (size[0] - text_width) // 2
        y = (size[1] - text_height) // 2
        
        # Draw border (multiple rectangles for thickness)
        border_color = (180, 0, 0, 255)  # Dark red
        for i in range(border_thickness):
            draw.rectangle([i, i, size[0]-1-i, size[1]-1-i], 
                         outline=border_color, width=1)
        
        # Draw text
        text_color = (200, 0, 0, 255)  # Red
        draw.text((x, y), text, fill=text_color, font=font)
        
        # Rotate the stamp
        if rotation != 0:
            img = img.rotate(rotation, expand=True, fillcolor=(0, 0, 0, 0))
        
        return img
    
    def apply_worn_effect(self, stamp_img, wear_intensity=0.3):
        """
        Apply a worn/aged effect to the stamp
        """
        # Convert PIL to numpy array
        stamp_array = np.array(stamp_img)
        
        # Create noise for worn effect
        noise = np.random.random(stamp_array.shape[:2])
        
        # Apply wear to alpha channel
        if stamp_array.shape[2] == 4:  # RGBA
            alpha = stamp_array[:, :, 3].astype(float) / 255.0
            
            # Reduce alpha where noise is high
            worn_mask = noise > (1 - wear_intensity)
            alpha[worn_mask] *= np.random.uniform(0.1, 0.7, np.sum(worn_mask))
            
            # Add some random spots
            spots = np.random.random(stamp_array.shape[:2]) > 0.98
            alpha[spots] *= 0.2
            
            stamp_array[:, :, 3] = (alpha * 255).astype(np.uint8)
        
        return Image.fromarray(stamp_array)
    
    def stamp_on_image(self, background_path, stamp_img, position, 
                      opacity=0.7, blend_mode='multiply'):
        """
        Apply stamp effect on an image
        """
        # Load background image
        background = Image.open(background_path).convert('RGBA')
        
        # Resize stamp if needed
        stamp_resized = stamp_img.copy()
        
        # Adjust stamp opacity
        if stamp_resized.mode == 'RGBA':
            # Modify alpha channel for opacity
            stamp_array = np.array(stamp_resized)
            stamp_array[:, :, 3] = (stamp_array[:, :, 3] * opacity).astype(np.uint8)
            stamp_resized = Image.fromarray(stamp_array)
        
        # Create a new layer for blending
        stamp_layer = Image.new('RGBA', background.size, (0, 0, 0, 0))
        stamp_layer.paste(stamp_resized, position, stamp_resized)
        
        # Blend the stamp with background
        if blend_mode == 'multiply':
            # Convert to RGB for multiply blend
            bg_rgb = background.convert('RGB')
            stamp_rgb = stamp_layer.convert('RGB')
            
            # Multiply blend
            bg_array = np.array(bg_rgb, dtype=np.float32) / 255.0
            stamp_array = np.array(stamp_rgb, dtype=np.float32) / 255.0
            
            # Only apply where stamp has content
            mask = np.array(stamp_layer)[:, :, 3] > 0
            result_array = bg_array.copy()
            
            for c in range(3):  # RGB channels
                result_array[:, :, c][mask] = (
                    bg_array[:, :, c][mask] * stamp_array[:, :, c][mask]
                )
            
            result = Image.fromarray((result_array * 255).astype(np.uint8))
        else:
            # Simple alpha composite
            result = Image.alpha_composite(background, stamp_layer)
        
        return result.convert('RGB')
    
    def create_ink_stamp_effect(self, image_path, text, 
                               position=(50, 50), color=(180, 0, 0)):
        """
        Create an ink stamp effect on an image
        """
        # Load image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create stamp texture
        h, w = img_rgb.shape[:2]
        stamp_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        # Position text
        x, y = position
        cv2.putText(stamp_mask, text, (x, y + text_h), 
                   font, font_scale, 255, thickness)
        
        # Create border around text
        border_size = 20
        x1, y1 = max(0, x - border_size), max(0, y - border_size)
        x2, y2 = min(w, x + text_w + border_size), min(h, y + text_h + border_size)
        cv2.rectangle(stamp_mask, (x1, y1), (x2, y2), 255, 3)
        
        # Add noise and imperfections
        noise = np.random.randint(0, 50, (h, w), dtype=np.uint8)
        stamp_mask = cv2.subtract(stamp_mask, noise)
        
        # Apply Gaussian blur for softer edges
        stamp_mask = cv2.GaussianBlur(stamp_mask, (3, 3), 0)
        
        # Create colored stamp
        stamp_colored = np.zeros_like(img_rgb)
        stamp_colored[:, :] = color
        
        # Blend with original image
        stamp_alpha = stamp_mask.astype(np.float32) / 255.0
        stamp_alpha = np.stack([stamp_alpha] * 3, axis=2)
        
        result = img_rgb.astype(np.float32)
        stamp_effect = stamp_colored.astype(np.float32) * stamp_alpha * 0.7
        
        result = result * (1 - stamp_alpha * 0.7) + stamp_effect
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result

# Example usage
def demo_stamping_effects():
    """
    Demonstrate various stamping effects
    """
    stamper = StampEffect()
    
    # Create a rubber stamp
    print("Creating rubber stamp...")
    stamp = stamper.create_rubber_stamp("APPROVED", size=(150, 150), rotation=20)
    
    # Apply worn effect
    worn_stamp = stamper.apply_worn_effect(stamp, wear_intensity=0.4)
    
    # Save stamps
    stamp.save('rubber_stamp.png')
    worn_stamp.save('worn_stamp.png')
    print("Stamps saved as 'rubber_stamp.png' and 'worn_stamp.png'")
    
    # Note: For applying stamps to images, you would need actual image files
    # Example of how to use:
    # result = stamper.stamp_on_image('document.jpg', worn_stamp, (100, 100))
    # result.save('stamped_document.jpg')
    
    # Apply stamp to sample document
    result = stamper.stamp_on_image('tests/input/6/target.png', worn_stamp, (200, 150))
    result.save('tests/output/6/target_stamp.png')
    print("Sample stamped document saved as 'tests/output/6/target_stamp.png'")

if __name__ == "__main__":
    demo_stamping_effects()