
import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any
import torch

class GeniesRGBToHSVNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    CATEGORY = "image"
    RETURN_TYPES = ("MASK", "IMAGE",)
    RETURN_NAMES = ("V Channel", "RGB Image",)
    FUNCTION = "get_v_channel"

    def get_v_channel(self, image):
        print(f"!!!!!!!!!!!!!!!!! input image shape: {image.shape}, type: {type(image)}")
        # Ensure the input is in the correct format: [Batch, Channels, Height, Width]
        if image.dim() != 4 or image.shape[3] != 3:
            raise ValueError("Input image must be a 4D tensor with shape [Batch, Channels=3, Height, Width]")

        images_np = image.cpu().numpy()

        v_channels = []
        rgb_images = []
        for image_np in images_np:
            hsv_image = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            v_channel = hsv_image[:, :, 2]
            v_channels.append(v_channel)

            # Create an HSV image with fixed H and S values (0, 0) and varying V
            rgb_image = cv2.cvtColor(np.stack([np.zeros_like(v_channel),  # H = 0
                                            np.zeros_like(v_channel),  # S = 0
                                            v_channel], axis=-1),      # V channel
                                    cv2.COLOR_HSV2RGB)
            rgb_images.append(rgb_image)

        # Stack V channels and normalize to [0, 1]
        v_channels_np = np.stack(v_channels, axis=0)  # Shape: [Batch, Height, Width]
        v_channels_normalized = v_channels_np.astype(np.float32) / 255.0

        # Convert back to torch tensor with shape [Batch, Height, Width, 1]
        v_channels_tensor = torch.tensor(v_channels_normalized).unsqueeze(-1)  # Shape: [Batch, Height, Width, 1]

        # Convert RGB images to torch tensor (optional step)
        rgb_images_tensor = torch.tensor(np.stack(rgb_images, axis=0) / 255.0).float()  # Shape: [Batch, Height, Width, Channels=3]
        print(f"!!!!!!!!!!!!!!!!! v_channels_tensor shape: {v_channels_tensor.shape}, type: {type(v_channels_tensor)}")
        print(f"!!!!!!!!!!!!!!!!! rgb_images_tensor shape: {rgb_images_tensor.shape}, type: {type(rgb_images_tensor)}")
        return (v_channels_tensor, rgb_images_tensor,)

class GeniesSelectRGBByMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }           
        
    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Selected Image",)
    FUNCTION = "select_rgb_by_mask"
    
    def select_rgb_by_mask(self, image, mask):
        print(f"!!!!!!!!!!!!!!!!! image shape: {image.shape}, type: {type(image)}")
        print(f"!!!!!!!!!!!!!!!!! mask shape: {mask.shape}, type: {type(mask)}")
        if image.shape[3] != 3:
            image = image[:, :, :, :3]
        # return error if image's size is different from mask's size
        if image.shape[1] != mask.shape[1] or image.shape[2] != mask.shape[2]:
            raise ValueError("Image and mask must have the same height and width")
        
        # if mask is not 0,1 mask, convert it to 0,1 mask, we only want 0 and 1 in the mask
        if mask.max() > 1:
            mask = mask / 255.0

        # convert mask to 0,1 mask for all 3 rgb channels
        mask = mask.unsqueeze(-1).expand(-1, -1, -1, 3)
        print(f"!!!!!!!!!!!!!!!!! mask shape: {mask.shape}, type: {type(mask)}")
        
        return (image * mask,)
    
