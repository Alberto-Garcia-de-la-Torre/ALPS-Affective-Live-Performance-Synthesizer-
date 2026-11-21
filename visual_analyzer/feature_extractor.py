import json
import csv
import math
from pathlib import Path

# Indici MediaPipe Pose utili
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_ANKLE, R_ANKLE = 27, 28
NOSE = 0


def _dist(a, b):
    """Distanza euclidea 2D tra due landmark (x, y)."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle(a, b, c):
    """Angolo in gradi nel punto b, tra i segmenti b-a e b-c."""
    ax, ay = a[0] - b[0], a[1] - b[1]
    cx, cy = c[0] - b[0], c[1] - b[1]
    dot = ax * cx + ay * cy
    mag = math.hypot(ax, ay) * math.hypot(cx, cy)
    if mag == 0:
        return 0.0
    cos_val = max(-1.0, min(1.0, dot / mag))
    return math.degrees(math.acos(cos_val))


def extract_features(frames_data, fps):
    """
    Calcola le metriche frame-by-frame a partire dai landmark grezzi.
    Restituisce una lista di dict, uno per frame con posa valida.
    """
    features = []
    prev_wrists = None  # per la velocità
    prev_ts = None

    for entry in frames_data:
        lm = entry["landmarks"]
        if lm is None:
            prev_wrists = None  # resetta la velocità se manca la posa
            prev_ts = None
            continue

        ts = entry["timestamp_s"]

        # 1. Distanza polso-spalla (apertura del gesto, media DX/SX)
        d_l = _dist(lm[L_WRIST], lm[L_SHOULDER])
        d_r = _dist(lm[R_WRIST], lm[R_SHOULDER])
        wrist_shoulder_dist = (d_l + d_r) / 2.0

        # 2. Angoli dei gomiti (postura braccio)
        ang_l = _angle(lm[L_SHOULDER], lm[L_ELBOW], lm[L_WRIST])
        ang_r = _angle(lm[R_SHOULDER], lm[R_ELBOW], lm[R_WRIST])

        # 3. Estensione verticale (testa-caviglie, media)
        ext_l = abs(lm[NOSE][1] - lm[L_ANKLE][1])
        ext_r = abs(lm[NOSE][1] - lm[R_ANKLE][1])
        vertical_extent = (ext_l + ext_r) / 2.0

        # 4. Velocità dei polsi (normalizzata per il tempo)
        if prev_wrists is not None and prev_ts is not None and ts > prev_ts:
            dt = ts - prev_ts
            v_l = _dist(lm[L_WRIST], prev_wrists[0]) / dt
            v_r = _dist(lm[R_WRIST], prev_wrists[1]) / dt
            wrist_speed = (v_l + v_r) / 2.0
        else:
            wrist_speed = 0.0

        features.append({
            "frame": entry["frame"],
            "timestamp_s": ts,
            "wrist_shoulder_dist": wrist_shoulder_dist,
            "elbow_angle_left": ang_l,
            "elbow_angle_right": ang_r,
            "vertical_extent": vertical_extent,
            "wrist_speed": wrist_speed,
        })

        prev_wrists = (lm[L_WRIST], lm[R_WRIST])
        prev_ts = ts

    return features


def save_landmarks_json(frames_data, output_path):
    """Salva i landmark grezzi in JSON (utile per ricalcoli futuri)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(frames_data, f)
    print(f"[feature_extractor] Landmark salvati in {output_path}")


def save_features_csv(features, output_path):
    """Salva le metriche derivate in CSV (pronto per pandas/plot)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not features:
        print("[feature_extractor] Nessuna feature da salvare.")
        return
    fieldnames = list(features[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(features)
    print(f"[feature_extractor] {len(features)} feature salvate in {output_path}")