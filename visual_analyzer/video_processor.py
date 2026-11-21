import cv2
from .pose_detector import PoseDetector

class VideoProcessor:

    def __init__(self, video_path, detector: PoseDetector = None):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError("Unable to open video")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        self.detector = detector

    def get_info(self):
        return {
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "total_frames": self.total_frames,
            "duration": self.duration
        }

    def process_video(self):
        """Elabora il video e restituisce i landmark per ogni frame."""
        frame_number = 0
        frames_data = []

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            timestamp = frame_number / self.fps
            timestamp_ms = int(timestamp * 1000)

            # Se il detector è stato passato, estrai i landmark
            landmarks = None
            if self.detector:
                landmarks = self.detector.detect(frame, timestamp_ms)
            
            # Salva i dati (utile per feature_extractor.py)
            frames_data.append({
                "frame": frame_number,
                "timestamp_s": timestamp,
                "landmarks": landmarks
            })

            # Stampa di debug ogni 100 frame per non intasare il terminale
            if frame_number % 100 == 0:
                print(f"Processing frame {frame_number} | Time: {timestamp:.2f}s | Pose: {'OK' if landmarks else 'N/A'}")

            frame_number += 1

        self.cap.release()
        print(f"Elaborazione completata. Totale frame: {frame_number}")
        return frames_data