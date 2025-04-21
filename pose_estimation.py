import base64
import json
from typing import List, Tuple, Union
import os

import boto3
import cv2
import numpy as np
from PIL import Image
import torch


class PoseEstimator:
    def __init__(self):
        pass

    def encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode an image file to base64.

        Args:
            image_path (str): The path to the image file.

        Returns:
            str: The base64-encoded image as a string.
        """
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_image


    def get_image_dimensions(self, image_path: str) -> Tuple[int, int]:
        """
        Get the dimensions (width and height) of an image.

        Args:
            image_path (str): The path to the image file.

        Returns:
            Tuple[int, int]: A tuple containing the width and height of the image.
        """

        with Image.open(image_path) as img:
            width, height = img.size
        return width, height


    def create_json_request(self,
        images: torch.Tensor,
        radius: int = 3,
        show_kpt_idx: bool = False,
        bounding_box: List[List[float]] = None,
    ) -> str:
        """
        Create a JSON structure containing base64 encoded data and sizes of images.

        Args:
            paths (Union[str, List[str]]): Either a single path string or a list of paths.
            raduis: The radius of the keypoint dot in visualization
            show_kpt_idx: switch on whether to show the labels for each keypoint on visualization
            font_size: font size of the keypoint labels in the visualization images
            bounding_box: list of bounding box for each image, each bounding is a list of 4 numbers
            [x1, y1, x2, y2]
        Returns:
            Dict[str, Dict]: Dictionary containing base64 encoded data and sizes of the images.
        """
        result = {
            "visualization_options": {
                "radius": radius,
                "show_kpt_idx": show_kpt_idx,
            },
            "input_images": {},
        }
        
        if len(images) != 0:
            for idx, image in enumerate(images):
                image_np = image.cpu().numpy()
                #save to file
                temp_image_path = f"temp_image_{idx}.jpg"
                cv2.imwrite(temp_image_path, image_np)
                encoded = self.encode_image_to_base64(temp_image_path)
                width, height = self.get_image_dimensions(temp_image_path)
                # remove the temp image
                os.remove(temp_image_path)
                result["input_images"][temp_image_path] = {
                    "image": encoded,
                    "size": {"width": width, "height": height},
                }
                if bounding_box:
                    result["input_images"][temp_image_path]["bounding_box"] = bounding_box[idx]
        # Dump data to the specified JSON file
        return json.dumps(result, indent=4)


class GeniesPoseEstimation:
    def __init__(self):
        self.pose_estimator = PoseEstimator()
        self.runtime_sm_client = boto3.client(service_name="sagemaker-runtime")
        self.endpoint_names = {
            "face": "facial-landmark-endpoint",
            "body": "body-pose-estimation-endpoint",
        }
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bbox_mask": ("MASK",),
                "models": ("STRING", {"default": "face", "choices": ["face", "body"]}),
            },
        }
    
    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("Keypoint Image", "Pose Estimation Image", "keypoints string")
    FUNCTION = "pose_estimation"

    def pose_estimation(self, image, bbox_mask,models):
        print(f"image shape: {image.shape}")
        bbox_masks = bbox_mask.cpu().numpy()
        bboxes = self._get_bbox_from_masks(bbox_masks)
        print(f"bboxes: {bboxes}")
        
        
        json_request = self.pose_estimator.create_json_request(image, bounding_box=bboxes)
        response = self.runtime_sm_client.invoke_endpoint(
            EndpointName=self.endpoint_names[models],
            Body=json_request,
            ContentType="application/json",
        )
        response = json.loads(response["Body"].read().decode("utf-8"))
        # print(f"response: {response}")
        status_code = response["code"]
        result_1 = json.loads(response["result"])
        status = result_1["status"]
        if status_code != 200 or status != "success":
            raise Exception(f"Error: {response['message']}")
        else:
            result_2 = result_1["data"]["result"]
            message = result_1["message"]
            for key, value in result_2.items():
                value = json.loads(value)
                predictions = value["predictions"][0][0]
                keypoints = predictions["keypoints"]
                scores = predictions["keypoint_scores"]
                bbox = predictions["bbox"]
        
            # plot the keypoints on an empty image
            empty_image = np.zeros((image.shape[1], image.shape[2], 3), dtype=np.uint8)
            image_np = image[0].cpu().numpy()
            keypoints_image = self._plot_keypoints(empty_image, keypoints, bboxes[0])
            pose_estimation_image = self._plot_keypoints(image_np, keypoints, bboxes[0])
            return keypoints_image, pose_estimation_image, keypoints

    def _plot_keypoints(self, image, keypoints, bbox=None):
        # plot the keypoints on an empty image
        result_image = image.copy()
        for keypoint in keypoints:
            result_image = cv2.circle(result_image, (int(keypoint[0]), int(keypoint[1])), 5, (0, 0, 1), -1)
        if bbox:    
            result_image = cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 1), 2)
        #save the empty image
        # cv2.imwrite("empty_image.jpg", empty_image)
        # convert the empty image to torch.Tensor
        result_image = torch.from_numpy(result_image)
        #unsqueeze the empty image
        result_image = result_image.unsqueeze(0)
        return result_image
    
    def _get_bbox_from_masks(self, masks):
        # get the bounding box from the mask
        # mask is a numpy array of shape (H, W)
        # return a list of 4 numbers [x1, y1, x2, y2]
        bboxes = []
        for mask in masks:
            print(f"mask shape: {mask.shape}")
            mask = mask.transpose()
            xx, yy = np.where(mask == 1)
            x1 = int(np.min(xx))
            y1 = int(np.min(yy)) - 10
            x2 = int(np.max(xx))
            y2 = int(np.max(yy)) + 10
            bboxes.append([x1, y1, x2, y2])
        return bboxes

class GeniesScaleFaceByKeypoints:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_src": ("IMAGE",),
                "image_tgt": ("IMAGE",),
                "keypoints_src": ("STRING",),
                "keypoints_tgt": ("STRING",)
            },
        }
        
    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Scaled IMAGE",)
    FUNCTION = "scale_face_by_keypoints"

    def scale_face_by_keypoints(self, image_src, image_tgt, keypoints_src, keypoints_tgt):
        print("max value of image_src: ", image_src.max())
        if image_src.max() <= 1:
            image_src = image_src * 255.0
        if image_tgt.max() <= 1:
            image_tgt = image_tgt * 255.0
        # Process all images in the batch
        batch_size = image_src.shape[0]
        print(f"Processing batch of {batch_size} images")
        
        # Create a list to store the results for each batch item
        result_tensors = []
        
        # Process each image in the batch
        for batch_idx in range(batch_size):
            # Extract single image from batch
            current_img_src = image_src[batch_idx].cpu().numpy()
            current_img_tgt = image_tgt[batch_idx].cpu().numpy()
            
            print(f"Image {batch_idx} - src shape: {current_img_src.shape}, tgt shape: {current_img_tgt.shape}")
            
            # Convert string keypoints to arrays if they're passed as strings

            current_keypoints_src = keypoints_src
            current_keypoints_tgt = keypoints_tgt
        
            if isinstance(current_keypoints_src, str):
                current_keypoints_src = json.loads(current_keypoints_src)
            if isinstance(current_keypoints_tgt, str):
                current_keypoints_tgt = json.loads(current_keypoints_tgt)
                
            tgt_size = current_img_tgt.shape[0], current_img_tgt.shape[1]
            
            # Get face width from keypoints
            width_src, _ = self._get_keypoints_bbox_dimensions(current_keypoints_src)
            width_tgt, _ = self._get_keypoints_bbox_dimensions(current_keypoints_tgt)
            print(f"Batch {batch_idx} - width_src: {width_src}, width_tgt: {width_tgt}")

                
            # Calculate scale factor based on face width ratio
            scale_factor = width_tgt / width_src
            print(f"Batch {batch_idx} - scale_factor: {scale_factor}")
            
            # Resize the source image
            image_scaled = cv2.resize(current_img_src, None, fx=scale_factor, fy=scale_factor)
            
            # Create a new canvas with the target size dimensions
            if isinstance(tgt_size, tuple) and len(tgt_size) == 2:
                new_height, new_width = tgt_size
            else:
                # Default size
                new_height, new_width = 1024, 1024
            
            # Detect background color from the corners of the scaled image
            h, w = image_scaled.shape[:2]
            corners = [
                image_scaled[0, 0],      # top-left
                image_scaled[0, w-1],    # top-right
                image_scaled[h-1, 0],    # bottom-left
                image_scaled[h-1, w-1]   # bottom-right
            ]
            bg_color = np.mean(corners, axis=0).astype(np.uint8)
            
            # Create new image filled with the background color
            new_image = np.ones((new_height, new_width, 3), dtype=np.uint8) * bg_color
            
            # Get nose keypoint positions (index 30)
            nose_src = np.array(current_keypoints_src[30])
            nose_tgt = np.array(current_keypoints_tgt[30])
            
            # Find where the nose will be after scaling
            nose_src_in_scaled = nose_src * scale_factor
            
            # Calculate the offset needed to align the noses
            offset_x = int(nose_tgt[0] - nose_src_in_scaled[0])
            offset_y = int(nose_tgt[1] - nose_src_in_scaled[1])
            
            # Determine the region where the scaled image will be placed
            x_start = max(0, offset_x)
            y_start = max(0, offset_y)
            
            # Determine the region of the scaled image to use
            img_x_start = max(0, -offset_x)
            img_y_start = max(0, -offset_y)
            
            # Determine the width and height to copy
            width_to_copy = min(image_scaled.shape[1] - img_x_start, new_width - x_start)
            height_to_copy = min(image_scaled.shape[0] - img_y_start, new_height - y_start)
            
            # Copy the scaled image to the new canvas
            if width_to_copy > 0 and height_to_copy > 0:
                new_image[y_start:y_start+height_to_copy, x_start:x_start+width_to_copy] = \
                    image_scaled[img_y_start:img_y_start+height_to_copy, img_x_start:img_x_start+width_to_copy]
            
            # Convert the new image to torch tensor (keep HWC format)
            new_image_tensor = torch.from_numpy(new_image)
            result_tensors.append(new_image_tensor)
        
        # Stack all tensors in the batch dimension
        final_tensor = torch.stack(result_tensors, dim=0)
        final_tensor = final_tensor.to(torch.float32) / 255.0
        print(f"Final tensor shape: {final_tensor.shape}")
        return (final_tensor,)
    
    def _get_keypoints_bbox_dimensions(self, keypoints):
        keypoints = np.array(keypoints)
        x_min = np.min(keypoints[:, 0])
        x_max = np.max(keypoints[:, 0])
        y_min = np.min(keypoints[:, 1])
        y_max = np.max(keypoints[:, 1])
        width = x_max - x_min
        height = y_max - y_min
        return width, height




if __name__ == "__main__":
    # test 
    pose_estimator = GeniesPoseEstimation()


