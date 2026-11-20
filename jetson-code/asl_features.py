"""Shared keypoint extractor: MediaPipe Pose + Hands, NO face mesh.

This ONE file is used by the training notebook AND by the Jetson (jetson-code/), so the features
the model is trained on are exactly the features it sees live.

Two steps:
1. Raw keypoints (`extract_keypoints`, `Detector.process`): MediaPipe's own coordinates, 0..1 across
   the picture. Used for the preview skeleton, the hand gate and the training notebook's hand check.
2. Model features (`normalize`): the raw keypoints re-expressed relative to the body, so distance to
   the camera, position in the picture, picture shape and sideways lean do not change them.

Both are FEATURE_DIM = 258 values per frame, in this order:
    pose        33 landmarks x 4                        132   [0:132]
    hand slot A 21 landmarks x 3                         63   [132:195]  MediaPipe label "Left"
    hand slot B 21 landmarks x 3                         63   [195:258]  MediaPipe label "Right"
Raw: pose (x, y, z, visibility), hands (x, y, z). A part that is not detected is all zeros.

Model features (FEATURE_VERSION "pose-hands-v2"), see `normalize`:
    pose  (x, y, z, visibility) with x, y relative to the shoulder midpoint, divided by the shoulder
          width and rotated so the shoulder line is level. All 33 points are kept (signs can touch
          the leg or other body parts), but a point with visibility below 0.5 has x, y, z set to 0,
          because MediaPipe guesses points it cannot see.
    hand  [wrist x, wrist y in the same body frame, hand size / shoulder width] followed by the other
          20 points relative to the wrist, divided by the hand size (same rotation).
x is multiplied by width / height first, because MediaPipe scales x by the picture width and y by its
height; without this a portrait and a landscape video of the same sign would give different numbers.

Hand slots: MediaPipe Hands assumes a mirrored (selfie) image. We feed it UNMIRRORED frames
(laptop webcam while training, glasses camera on the Jetson), so its "Left" label is the hand that
appears on the image's LEFT, i.e. the signer's right hand. The labels are only used to keep each hand
in a consistent slot; both training and the Jetson see the same unmirrored geometry, so they agree.
Never flip frames before extracting.

Compatible with Python 3.8+ and MediaPipe 0.10.x (mp.solutions API).
"""
from typing import Optional, Tuple

import numpy as np

FEATURE_VERSION = "pose-hands-v2"

POSE_COUNT = 33
HAND_COUNT = 21
POSE_DIMS = POSE_COUNT * 4          # 132
HAND_DIMS = HAND_COUNT * 3          # 63
FEATURE_DIM = POSE_DIMS + 2 * HAND_DIMS  # 258

POSE_SLICE = slice(0, POSE_DIMS)
HAND_A_SLICE = slice(POSE_DIMS, POSE_DIMS + HAND_DIMS)
HAND_B_SLICE = slice(POSE_DIMS + HAND_DIMS, FEATURE_DIM)
HANDS_SLICE = slice(POSE_DIMS, FEATURE_DIM)

L_SHOULDER, R_SHOULDER = 11, 12
WRIST, MIDDLE_MCP = 0, 9
DEFAULT_ASPECT = 16 / 9
NO_POSE_SCALE = 0.3   # body scale (fraction of picture height) used when no pose is detected
FEATURE_CLIP = 10.0
POSE_VISIBILITY_MIN = 0.5  # pose points below this visibility are zeroed in the model features


def _hand_array(hand_landmarks) -> np.ndarray:
    return np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark], dtype=np.float32).flatten()


def _pose_array(pose_landmarks) -> np.ndarray:
    return np.array(
        [[p.x, p.y, p.z, p.visibility] for p in pose_landmarks.landmark], dtype=np.float32
    ).flatten()


def assign_hands(multi_hand_landmarks, multi_handedness) -> Tuple[Optional[object], Optional[object]]:
    """Return (slot_A_hand, slot_B_hand) from a MediaPipe Hands result. Either may be None."""
    if not multi_hand_landmarks:
        return None, None
    hands = []
    for i, lm in enumerate(multi_hand_landmarks):
        label, score = "Left", 0.0
        if multi_handedness and i < len(multi_handedness):
            top = multi_handedness[i].classification[0]
            label, score = top.label, float(top.score)
        hands.append({"lm": lm, "label": label, "score": score, "x": float(lm.landmark[0].x)})

    slot_a = [h for h in hands if h["label"] == "Left"]
    slot_b = [h for h in hands if h["label"] != "Left"]
    if len(slot_a) > 1 or len(slot_b) > 1:
        # Two hands got the same label (a misclassification): fall back to image position,
        # the hand further left in the image goes to slot A.
        hands.sort(key=lambda h: h["x"])
        if len(hands) == 1:
            return hands[0]["lm"], None
        return hands[0]["lm"], hands[1]["lm"]
    return (slot_a[0]["lm"] if slot_a else None), (slot_b[0]["lm"] if slot_b else None)


def extract_keypoints(pose_result, hands_result) -> np.ndarray:
    """Raw 258-value keypoint vector (picture coordinates) from MediaPipe Pose and Hands results.
    Pass it through `normalize` before giving it to the model."""
    pose = (
        _pose_array(pose_result.pose_landmarks)
        if pose_result.pose_landmarks
        else np.zeros(POSE_DIMS, dtype=np.float32)
    )
    a, b = assign_hands(hands_result.multi_hand_landmarks, hands_result.multi_handedness)
    hand_a = _hand_array(a) if a is not None else np.zeros(HAND_DIMS, dtype=np.float32)
    hand_b = _hand_array(b) if b is not None else np.zeros(HAND_DIMS, dtype=np.float32)
    return np.concatenate([pose, hand_a, hand_b]).astype(np.float32)


def hands_present(features: np.ndarray) -> bool:
    """True if at least one hand was detected (works on raw keypoints and on model features)."""
    return bool(np.any(features[HANDS_SLICE] != 0))


def normalize(raw: np.ndarray, aspect: float = DEFAULT_ASPECT) -> np.ndarray:
    """Raw keypoints -> model features (FEATURE_VERSION). Accepts one frame (258,) or frames (T, 258).

    aspect: width / height of the picture the keypoints came from.
    """
    raw = np.asarray(raw, dtype=np.float32)
    frames = raw.reshape(-1, FEATURE_DIM)
    n = frames.shape[0]
    pose = frames[:, POSE_SLICE].reshape(n, POSE_COUNT, 4)
    hands = (frames[:, HAND_A_SLICE].reshape(n, HAND_COUNT, 3), frames[:, HAND_B_SLICE].reshape(n, HAND_COUNT, 3))

    pose_ok = np.any(pose != 0, axis=(1, 2))
    pose_xy = pose[:, :, :2] * np.array([aspect, 1.0], dtype=np.float32)
    shoulder_vec = pose_xy[:, L_SHOULDER] - pose_xy[:, R_SHOULDER]
    width = np.linalg.norm(shoulder_vec, axis=1)
    body_ok = pose_ok & (width > 1e-3)
    center = np.where(body_ok[:, None], (pose_xy[:, L_SHOULDER] + pose_xy[:, R_SHOULDER]) / 2,
                      np.array([0.5 * aspect, 0.5], dtype=np.float32))
    scale = np.where(body_ok, width, NO_POSE_SCALE).astype(np.float32)
    angle = np.where(body_ok, np.arctan2(shoulder_vec[:, 1], shoulder_vec[:, 0]), 0.0)
    # Level the shoulder LINE, not its direction, so a mirrored person is not turned upside down.
    angle = np.where(angle > np.pi / 2, angle - np.pi, np.where(angle < -np.pi / 2, angle + np.pi, angle))
    cos, sin = np.cos(-angle)[:, None], np.sin(-angle)[:, None]

    def rotate(v):  # (n, k, 2)
        return np.stack([cos * v[..., 0] - sin * v[..., 1], sin * v[..., 0] + cos * v[..., 1]], axis=-1)

    out = np.zeros_like(frames)
    pose_out = np.zeros((n, POSE_COUNT, 4), dtype=np.float32)
    pose_out[:, :, :2] = rotate((pose_xy - center[:, None]) / scale[:, None, None])
    pose_out[:, :, 2] = pose[:, :, 2] * aspect / scale[:, None]
    pose_out[:, :, 3] = pose[:, :, 3]
    # MediaPipe guesses points it cannot see (legs below the picture, for example): drop those
    # positions so upper-body and full-body framing look the same where it matters.
    pose_out[:, :, :3] *= (pose[:, :, 3] >= POSE_VISIBILITY_MIN)[..., None]
    pose_out[~pose_ok] = 0
    out[:, POSE_SLICE] = pose_out.reshape(n, POSE_DIMS)

    for hand, part in zip(hands, (HAND_A_SLICE, HAND_B_SLICE)):
        ok = np.any(hand != 0, axis=(1, 2))
        xy = hand[:, :, :2] * np.array([aspect, 1.0], dtype=np.float32)
        size = np.maximum(np.linalg.norm(xy[:, MIDDLE_MCP] - xy[:, WRIST], axis=1), 1e-3)
        wrist = rotate((xy[:, WRIST:WRIST + 1] - center[:, None]) / scale[:, None, None])[:, 0]
        rel = rotate((xy[:, 1:] - xy[:, WRIST:WRIST + 1]) / size[:, None, None])
        rel_z = (hand[:, 1:, 2] - hand[:, WRIST:WRIST + 1, 2]) * aspect / size[:, None]
        hand_out = np.concatenate(
            [wrist, (size / scale)[:, None], np.concatenate([rel, rel_z[..., None]], axis=-1).reshape(n, -1)],
            axis=1,
        )
        hand_out[~ok] = 0
        out[:, part] = hand_out

    out = np.clip(np.nan_to_num(out), -FEATURE_CLIP, FEATURE_CLIP).astype(np.float32)
    return out.reshape(raw.shape)


class Detector:
    """Pose + Hands, configured identically for training and the Jetson.

    Use a NEW Detector for each video clip (it tracks between frames), and feed RGB frames in time order.
    """

    def __init__(
        self,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        import mediapipe as mp  # imported here so this file can be inspected without MediaPipe

        self._init_args = (model_complexity, min_detection_confidence, min_tracking_confidence)
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, rgb: np.ndarray) -> np.ndarray:
        """RGB uint8 frame (H, W, 3) -> raw keypoints (FEATURE_DIM,). Model features:
        normalize(raw, aspect=W / H)."""
        rgb.flags.writeable = False
        pose_result = self._pose.process(rgb)
        hands_result = self._hands.process(rgb)
        rgb.flags.writeable = True
        return extract_keypoints(pose_result, hands_result)

    def reset(self) -> None:
        """Forget tracking state (call between separate recordings on the Jetson)."""
        self.close()
        self.__init__(*self._init_args)

    def close(self) -> None:
        for solution in (self._pose, self._hands):
            try:
                solution.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
