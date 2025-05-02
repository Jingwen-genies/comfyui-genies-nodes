from . import pose_estimation
from . import image_process
from . import io_nodes

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeniesPoseEstimation": "Genies Pose Estimation",
    "GeniesScaleFaceByKeypoints": "Genies Scale Face by Keypoints",
    "GeniesRGBToHSV": "Get V Channel from HSV",
    "GeniesSelectRGBByMask": "Select RGB by Mask",
    "GeniesSaveImageToS3": "Save to s3",
    "GeniesGetImageFromS3": "Download from s3"
}

NODE_CLASS_MAPPINGS = {
    "GeniesPoseEstimation": pose_estimation.GeniesPoseEstimation,
    "GeniesScaleFaceByKeypoints": pose_estimation.GeniesScaleFaceByKeypoints,
    "GeniesRGBToHSV": image_process.GeniesRGBToHSVNode,
    "GeniesSelectRGBByMask": image_process.GeniesSelectRGBByMask,
    "GeniesSaveImageToS3":io_nodes.S3ImageSaver,
    "GeniesGetImageFromS3": io_nodes.S3ImageLoader


}