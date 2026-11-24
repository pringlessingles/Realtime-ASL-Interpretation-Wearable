r"""Test an exported ASL model with a local webcam (Python 3.10 or 3.11).

Examples from the repository root:
    .\.venv\Scripts\python.exe jetson-code/live_local.py --check
    .\.venv\Scripts\python.exe jetson-code/live_local.py
    .\.venv\Scripts\python.exe jetson-code/live_local.py --model jetson-code/models/asl_32word_test2/asl_32word_test2.h5
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model", type=Path,
        default=Path(__file__).resolve().parent / "models/asl_32word_test1/asl_32word_test1.h5",
        help="Exported .h5 file; model_meta.json must be in the same folder (default: test1).",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0; try 1 for another camera).")
    parser.add_argument("--check", action="store_true", help="Load and validate the model without opening the camera.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if sys.version_info[:2] not in ((3, 10), (3, 11)):
        raise RuntimeError("Use a Python 3.10 or 3.11 environment with requirements-local.txt.")
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not model_path.with_name("model_meta.json").is_file():
        raise FileNotFoundError(f"Keep model_meta.json next to {model_path.name}.")

    import cv2
    import numpy as np
    import tensorflow as tf

    from holistic_lstm_infer import FEATURE_DIM, HolisticLSTMInfer, load_model_meta, prepare_frame, validate_model

    meta = load_model_meta(str(model_path))
    actions = list(meta["actions"])
    window = int(meta["window"])
    frame_size = tuple(int(v) for v in meta["frame_size"])
    sample_fps = float(meta.get("effective_fps", meta["capture_fps"] / meta["frame_stride"]))
    if sample_fps <= 0:
        raise ValueError("The model's effective_fps must be positive.")
    model = tf.keras.models.load_model(str(model_path), compile=False)
    validate_model(model, window, actions)

    @tf.function(input_signature=[tf.TensorSpec([1, window, FEATURE_DIM], tf.float32)])
    def predict(x):
        return model(x, training=False)

    # Warm the graph and check the export before touching the camera. This checks
    # compatibility only; zeros are not an accuracy or sign-recognition test.
    probs = np.asarray(predict(np.zeros((1, window, FEATURE_DIM), dtype=np.float32)))
    if probs.shape != (1, len(actions)) or not np.isfinite(probs).all():
        raise RuntimeError(f"Invalid model output: {probs.shape}")
    print(f"Loaded {model_path}", flush=True)
    print(f"Input: {window} x {FEATURE_DIM}; outputs: {len(actions)}; sampling: {sample_fps:g} fps", flush=True)
    if args.check:
        print("Model load and inference check passed. Camera was not opened.")
        return 0

    cap = cv2.VideoCapture(args.camera)
    infer = None
    title = f"ASL laptop test - {model_path.stem}"
    try:
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam. Close other camera apps, check camera permissions, or try --camera 1.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        cap.set(cv2.CAP_PROP_FPS, float(meta["capture_fps"]))
        infer = HolisticLSTMInfer(
            model=model, actions=actions, sequence_length=window,
            threshold=float(meta["threshold"]), stable_n=int(meta["stable_n"]),
            model_complexity=int(meta["model_complexity"]),
            repeat_reset=int(meta["repeat_reset"]), min_hand_frac=float(meta["min_hand_frac"]),
            frame_size=frame_size, predict_fn=predict,
        )
        print("Keep your upper body and both hands visible. Press Q or Esc to quit.", flush=True)
        print("Frames are unmirrored, matching training. Predictions stay on this device.", flush=True)
        next_sample = 0.0
        last_sample = None
        measured_fps = None
        samples = 0
        latest_gloss = "--"
        latest_gloss_time = 0.0
        while True:
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError("The webcam stopped returning frames.")
            bgr = prepare_frame(bgr, frame_size)
            now = time.monotonic()
            if now >= next_sample:
                if last_sample is not None:
                    instant_fps = 1.0 / max(now - last_sample, 1e-6)
                    measured_fps = instant_fps if measured_fps is None else 0.9 * measured_fps + 0.1 * instant_fps
                last_sample = now
                next_sample = now + 1.0 / sample_fps
                # Sample by elapsed time rather than assuming the webcam delivers
                # 30 fps. Never mirror the frame before feature extraction.
                prediction = infer.push_frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                samples += 1
                if prediction is not None:
                    latest_gloss = f"{prediction.gloss} ({prediction.confidence:.0%})"
                    latest_gloss_time = time.monotonic()
                    print(f"[gloss] {latest_gloss}", flush=True)

            raw = "Waiting for enough hand frames"
            if samples < window:
                raw = f"Warming up: {samples}/{window} frames"
            elif infer.last_probs is not None:
                best = int(np.argmax(infer.last_probs))
                raw = f"Candidate: {actions[best]} ({infer.last_probs[best]:.0%})"
            stable = latest_gloss if time.monotonic() - latest_gloss_time < 5 else "--"
            rate = f"Sampling: {measured_fps:.1f}/{sample_fps:g} fps" if measured_fps else "Starting camera"
            lines = [model_path.stem, raw, f"Last accepted: {stable}", rate, "Q / Esc: quit"]
            if measured_fps is not None and measured_fps < sample_fps * 0.8:
                lines.append("Below target rate: live timing differs from training")
            for i, text in enumerate(lines):
                point = (12, 25 + i * 28)
                cv2.putText(bgr, text, point, cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(bgr, text, point, cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(title, bgr)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if infer is not None:
            infer.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
