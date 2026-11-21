from pathlib import Path
from video_processor import VideoProcessor

# Percorso assoluto della cartella principale del progetto
PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = PROJECT_ROOT / "videos" / "singer.mp4"

print("Video path:", VIDEO_PATH)
print("File exists:", VIDEO_PATH.exists())

video = VideoProcessor(str(VIDEO_PATH))

print("\nVIDEO INFORMATION")
print("-----------------")

for key, value in video.get_info().items():
    print(f"{key}: {value}")

print("\nSTARTING VIDEO PROCESSING\n")

video.process_video()

print("\nVIDEO PROCESSING COMPLETED")