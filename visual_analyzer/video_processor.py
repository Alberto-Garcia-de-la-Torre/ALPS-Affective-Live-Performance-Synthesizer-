
import cv2


class VideoProcessor:

    def __init__(self, video_path):

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError("Unable to open video")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.duration = self.total_frames / self.fps

    def get_info(self):

        return {
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "total_frames": self.total_frames,
            "duration": self.duration
        }

    def process_video(self):

        frame_number = 0

        while self.cap.isOpened():

            ret, frame = self.cap.read()

            if not ret:
                break

            timestamp = frame_number / self.fps

            print(f"Processing frame {frame_number} | Time: {timestamp:.2f}s")

            frame_number += 1

        self.cap.release()
