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
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def __init__(self):
        self.s3 = boto3.client('s3')
        
    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    def save_images(self, images: torch.Tensor, bucket_name: str, s3_key: str, prompt: Optional[str] = None) -> dict:
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
        for i, image in enumerate(images):
            try:
                # Convert tensor to numpy and scale to 0-255
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                
                # Add metadata if prompt provided
                metadata = None
                if prompt is not None:
                    metadata = PngInfo()
                    metadata.add_text("prompt", json.dumps(prompt))
                
                # Save image to bytes
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG', pnginfo=metadata)
                img_bytes.seek(0)
                
                # Generate unique key for each image
                key = s3_key.replace(".png", f"_{i:05d}.png")
                
                # Upload to S3
                self.s3.upload_fileobj(
                    img_bytes,
                    bucket_name,
                    key
                )
                
                results.append({
                    "filename": os.path.basename(key),
                    "subfolder": os.path.dirname(key),
                    "type": "output",
                    "s3_path": f"s3://{bucket_name}/{key}"
                })
                
            except Exception as e:
                print(f"Error saving image {i} to S3: {str(e)}")
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
    FUNCTION = "load_image_as_tensor"
    CATEGORY = "image"

    def __init__(self):
        """
        Initialize S3ImageLoader with AWS credentials and bucket name.
        
        Args:
            bucket_name (str): Name of the S3 bucket
            access_key (str): AWS access key ID
            secret_key (str): AWS secret access key
            region (str): AWS region (default: 'us-west-2')
        """
        self.s3 = boto3.client(
            's3',
        )

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    def load_image_as_tensor(self, key: str) -> Optional[torch.Tensor]:
        """
        Download an image from S3 and convert to tensor.
        
        Args:
            key (str): S3 key (path) of the image to download
            
        Returns:
            Optional[torch.Tensor]: Image tensor if successful, None otherwise
        """
        try:
            # Download image bytes from S3
            response = self.s3.get_object(
                Bucket=self.bucket_name,
                Key=key
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
            
            return image_tensor
        except Exception as e:
            print(f"Error loading image from S3: {str(e)}")
            return None


