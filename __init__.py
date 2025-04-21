from pose_estimation import *
from image_process import *

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeniesPoseEstimation": "Genies Pose Estimation",
    "GeniesScaleFaceByKeypoints": "Genies Scale Face by Keypoints",
    "GeniesRGBToHSV": "Get V Channel from HSV",
    "GeniesSelectRGBByMask": "Select RGB by Mask",
}

NODE_CLASS_MAPPINGS = {
    "GeniesPoseEstimation": GeniesPoseEstimation,
    "GeniesScaleFaceByKeypoints": GeniesScaleFaceByKeypoints,
    "GeniesRGBToHSV": GeniesRGBToHSVNode,
    "GeniesSelectRGBByMask": GeniesSelectRGBByMask,


}