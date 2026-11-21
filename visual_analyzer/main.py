from pathlib import Path
from video_processor import VideoProcessor
from pose_detector import PoseDetector

# Calcola la root del progetto (due livelli sopra visual_analyzer/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = PROJECT_ROOT / "videos" / "singer.mp4"
MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_full.task"

def main():
    print(f"Video: {VIDEO_PATH}")
    print(f"Modello: {MODEL_PATH}")

    # 1. Inizializza il rilevatore di pose
    detector = PoseDetector(MODEL_PATH)

    try:
        # 2. Inizializza il processore video passandogli il detector
        processor = VideoProcessor(VIDEO_PATH, detector=detector)
        
        # Stampa info video
        info = processor.get_info()
        print(f"Video Info: {info['width']}x{info['height']} @ {info['fps']:.2f} fps | Durata: {info['duration']:.2f}s")

        # 3. Elabora il video
        results = processor.process_video()

        # 4. Riepilogo
        valid_frames = [r for r in results if r["landmarks"] is not None]
        print(f"\nRiepilogo: {len(valid_frames)} frame con posa rilevata su {len(results)} totali.")

    finally:
        # Assicurati di chiudere sempre il detector
        detector.close()

if __name__ == "__main__":
    main()