import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ✅ Import ASSOLUTI completi (con il nome del package)
from visual_analyzer.video_processor import VideoProcessor
from visual_analyzer.pose_detector import PoseDetector
from visual_analyzer.feature_extractor import (
    extract_features, save_landmarks_json, save_features_csv
)

VIDEO_PATH = PROJECT_ROOT / "videos" / "singer.mp4"
MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_full.task"
OUTPUT_DIR = PROJECT_ROOT / "output"


def main():
    print(f"Video: {VIDEO_PATH}")
    print(f"Modello: {MODEL_PATH}")

    detector = PoseDetector(MODEL_PATH)
    try:
        processor = VideoProcessor(VIDEO_PATH, detector=detector)
        info = processor.get_info()
        print(f"Video Info: {info['width']}x{info['height']} @ {info['fps']:.2f} fps | "
              f"Durata: {info['duration']:.2f}s")

        results = processor.process_video()

        # --- Salvataggio landmark grezzi ---
        save_landmarks_json(results, OUTPUT_DIR / "landmarks.json")

        # --- Estrazione e salvataggio feature ---
        features = extract_features(results, info["fps"])
        save_features_csv(features, OUTPUT_DIR / "features.csv")

        print(f"\nRiepilogo: {len(features)} frame con feature calcolate "
              f"su {len(results)} totali.")
    finally:
        detector.close()


if __name__ == "__main__":
    main()