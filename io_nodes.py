import boto3
import os
import json
import base64
import requests
import backoff
import numpy as np
import torch
import io
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from typing import Optional, Tuple
import folder_paths
from comfy.model_management import get_torch_device
import datetime

class S3ImageSaver:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "bucket_name": ("STRING", {"default": "your-bucket-name"}),
                "s3_key": ("STRING", {"default": "outputs/image.png"}),
            },
            "optional": {
                "prompt": ("PROMPT",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "upload_images_to_s3"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def __init__(self):
        self.s3 = boto3.client('s3')
        
    def _process_s3_key(self, s3_key: str) -> tuple[str, str]:
        """
        Process s3_key to determine folder path and base filename.
        
        Args:
            s3_key (str): Input S3 key path
            
        Returns:
            tuple[str, str]: (folder_path, base_filename)
        """
        # Remove any leading/trailing slashes
        s3_key = s3_key.strip('/')
        
        # Check if it looks like a file (ends with .png)
        if s3_key.lower().endswith('.png'):
            # Use parent directory as folder and filename as base
            folder_path = os.path.dirname(s3_key)
            base_filename = os.path.splitext(os.path.basename(s3_key))[0]
        else:
            # If it doesn't end with a slash, treat it as a folder name
            if not s3_key.endswith('/'):
                s3_key = s3_key + '/'
            folder_path = s3_key
            base_filename = 'output'
            
        return folder_path, base_filename
        
    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    def upload_images_to_s3(self, images: torch.Tensor, bucket_name: str, s3_key: str, prompt: Optional[str] = None) -> dict:
        """
        Save a batch of images to S3.
        
        Args:
            images (torch.Tensor): Batch of images to save
            bucket_name (str): Name of the S3 bucket
            s3_key (str): Base S3 key (path) for the images
            prompt (Optional[str]): Prompt used to generate the images
            
        Returns:
            dict: UI information about saved images
        """
        results = []
        folder_path, base_filename = self._process_s3_key(s3_key)
        
        for i, image in enumerate(images):
            try:
                # Convert tensor to numpy and scale to 0-255
                image_np = image.cpu().numpy()
                image_np = (image_np * 255).astype(np.uint8)
                
                # Convert to PIL Image
                img = Image.fromarray(image_np)
                
                # Add metadata if prompt provided
                metadata = None
                if prompt is not None:
                    metadata = PngInfo()
                    metadata.add_text("prompt", json.dumps(prompt))
                
                # Save image to bytes
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG', pnginfo=metadata)
                img_bytes.seek(0)
                
                # Generate unique filename with timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{base_filename}_{timestamp}_{i:05d}.png"
                
                # Construct final key
                key = os.path.join(folder_path, filename) if folder_path else filename
                
                # Upload to S3
                self.s3.upload_fileobj(
                    img_bytes,
                    bucket_name,
                    key
                )
                print(f"saved image {i} to {bucket_name}, {key}")
                
                results.append({
                    "filename": filename,
                    "subfolder": folder_path,
                    "type": "output",
                    "s3_path": f"s3://{bucket_name}/{key}"
                })
                
            except Exception as e:
                print(f"Error saving image {i} to S3: {e}")
                continue

        return {"ui": {"images": results}}




class S3ImageLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bucket_name": ("STRING", {"default": "your-bucket-name"}),
                "s3_key": ("STRING", {"default": "path/to/image.png"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_image_from_s3"
    CATEGORY = "image"

    def __init__(self):
        self.s3 = boto3.client('s3')


    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    def load_image_from_s3(self, bucket_name:str, s3_key: str) -> Optional[torch.Tensor]:
        """
        Download an image from S3 and convert to tensor.
        
        Args:
            key (str): S3 key (path) of the image to download
            
        Returns:
            Optional[torch.Tensor]: Image tensor if successful, None otherwise
        """
        try:
            print(f"loading image from bucket: {bucket_name}, key: {s3_key}")
            # Download image bytes from S3
            response = self.s3.get_object(
                Bucket=bucket_name,
                Key=s3_key
            )
            image_bytes = response['Body'].read()
            
            # Convert to PIL Image
            img = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to numpy array and normalize
            image = np.array(img).astype(np.float32) / 255.0
            
            # Convert to tensor and add batch dimension
            image_tensor = torch.from_numpy(image)[None,]
            
            return (image_tensor,)  # 返回tuple
            
        except Exception as e:
            print(f"Error loading image from S3: {e}")
            raise Exception(f"Failed to load image from S3: {e}")


