import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path


class PoseDetector:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modello non trovato: {self.model_path}")

        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame, timestamp_ms):
        rgb_frame = frame[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            return [(lm.x, lm.y, lm.z, lm.visibility) for lm in result.pose_landmarks[0]]
        return None

    def close(self):
        self.detector.close()