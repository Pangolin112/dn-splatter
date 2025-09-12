import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import save_image

def resize_image(img, longest_edge):
    # resize to have the longest edge equal to longest_edge
    width, height = img.size
    if width > height:
        ratio = longest_edge / width
    else:
        ratio = longest_edge / height
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    return img.resize((new_width, new_height), Image.BILINEAR)

def interpolate_to_patch_size(img_bchw, patch_size):
    # Interpolate the image so that H and W are multiples of the patch size
    _, _, H, W = img_bchw.shape
    target_H = H // patch_size * patch_size
    target_W = W // patch_size * patch_size
    img_bchw = torch.nn.functional.interpolate(img_bchw, size=(target_H, target_W))
    return img_bchw, target_H, target_W

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dino_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    print("Loading DINOv2 model...")
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dinov2 = dinov2.to(device)

    image = Image.open("data/e9ac2fc517_original/DSC08479_original.png")
    image = resize_image(image, 512)
    image = dino_transform(image)[:3].unsqueeze(0) #[1, 3, 512, 512]
    print(image.shape)
    image, target_H, target_W = interpolate_to_patch_size(image, dinov2.patch_size)
    image = image.cuda()
    with torch.no_grad():
        features = dinov2.forward_features(image)["x_norm_patchtokens"][0]

    features = features.cpu()

    features_hwc = features.reshape((target_H // dinov2.patch_size, target_W // dinov2.patch_size, -1))
    features_chw = features_hwc.permute((2, 0, 1))
    print(features_chw.shape)  # [384, 36, 36]

if __name__ == "__main__":
    main()