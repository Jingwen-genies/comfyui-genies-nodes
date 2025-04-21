from . import pose_estimation
from . import image_process

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeniesPoseEstimation": "Genies Pose Estimation",
    "GeniesScaleFaceByKeypoints": "Genies Scale Face by Keypoints",
    "GeniesRGBToHSV": "Get V Channel from HSV",
    "GeniesSelectRGBByMask": "Select RGB by Mask",
}

NODE_CLASS_MAPPINGS = {
    "GeniesPoseEstimation": pose_estimation.GeniesPoseEstimation,
    "GeniesScaleFaceByKeypoints": pose_estimation.GeniesScaleFaceByKeypoints,
    "GeniesRGBToHSV": image_process.GeniesRGBToHSVNode,
    "GeniesSelectRGBByMask": image_process.GeniesSelectRGBByMask,


}