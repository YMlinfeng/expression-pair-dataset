"""MediaPipe FaceLandmarker extraction: 478 landmarks, 52 blendshapes, head pose.

One pass over a video yields everything the geometry-side metrics need. Blendshape
channels are always addressed *by name*: MediaPipe emits `_neutral` at index 0 and
omits `tongueOut`, so every ARKit-named slot is shifted by one relative to the
canonical ARKit-52 ordering.
"""

from __future__ import annotations

import math
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "face_landmarker.task"

# Iris centres exist only because FaceLandmarker v2 returns 478 (not 468) points.
LEFT_IRIS, RIGHT_IRIS = 468, 473
NOSE_TIP = 1
MOUTH_L, MOUTH_R = 61, 291
EYE_OUTER_L, EYE_OUTER_R = 33, 263

# Points chosen to stay put under expression change; they define the frame in
# which every other point's displacement is measured.
RIGID_IDS = [33, 133, 362, 263, 168, 6, 197, 195, 4, 1, 234, 454, 10]

REGIONS: dict[str, list[int]] = {
    "brow": [46, 53, 52, 65, 55, 70, 63, 105, 66, 107,
             285, 295, 282, 283, 276, 300, 293, 334, 296, 336],
    "eyelid": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
               263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
    "nose": [1, 2, 4, 5, 6, 19, 94, 97, 98, 115, 131,
             326, 327, 344, 360, 278, 279, 220, 440, 166, 392],
    "lip": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191],
    "jaw": [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234,
            377, 400, 378, 379, 365, 397, 288, 361, 323, 454],
}


def ensure_model(path: Path = MODEL_PATH) -> Path:
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, path)
    return path


@dataclass
class VideoFeats:
    """Per-frame features for one clip. Invalid frames are flagged in `ok`.

    `landmarks` are stored in pixel space, not MediaPipe's raw normalised space:
    raw x is divided by image width and y by image height, so on a non-square
    frame the two axes carry different units and every downstream distance is
    silently anisotropic. We rescale to (x*W, y*H, z*W) -- z is documented as
    sharing roughly the scale of x -- which makes the coordinates isotropic.
    """

    name: str
    ok: np.ndarray            # (T,) bool
    blendshapes: np.ndarray   # (T, 52) float32
    landmarks: np.ndarray     # (T, 478, 3) float32, isotropic pixel space
    pose: np.ndarray          # (T, 3) float32, degrees (yaw, pitch, roll)
    bs_names: list[str]
    width: int = 0
    height: int = 0
    fps: float = 25.0
    crop_box: tuple[int, int, int, int] | None = None
    frames_bgr: list[np.ndarray] | None = None

    def __len__(self) -> int:
        return int(self.ok.shape[0])

    @property
    def n_valid(self) -> int:
        return int(self.ok.sum())


def _matrix_to_euler(m: np.ndarray) -> tuple[float, float, float]:
    """Rotation matrix -> (yaw, pitch, roll) in degrees."""
    sy = math.sqrt(m[0, 0] ** 2 + m[1, 0] ** 2)
    if sy > 1e-6:
        roll = math.atan2(m[2, 1], m[2, 2])
        pitch = math.atan2(-m[2, 0], sy)
        yaw = math.atan2(m[1, 0], m[0, 0])
    else:
        roll = math.atan2(-m[1, 2], m[1, 1])
        pitch = math.atan2(-m[2, 0], sy)
        yaw = 0.0
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def read_video(path: str | Path, max_frames: int | None = None,
               resize_long: int | None = 512) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if resize_long:
            h, w = frame.shape[:2]
            scale = resize_long / max(h, w)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))),
                                   interpolation=cv2.INTER_AREA)
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded: {path}")
    return frames, float(fps)


def crop_frames(frames: list[np.ndarray], box: tuple[int, int, int, int],
                size: int) -> list[np.ndarray]:
    """Cut a fixed square box out of every frame, replicate-padding at the edges."""
    x0, y0, side, _ = box
    H, W = frames[0].shape[:2]
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x0 + side - W), max(0, y0 + side - H)
    out = []
    for f in frames:
        if pad_l or pad_t or pad_r or pad_b:
            f = cv2.copyMakeBorder(f, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
        chip = f[y0 + pad_t:y0 + pad_t + side, x0 + pad_l:x0 + pad_l + side]
        out.append(cv2.resize(chip, (size, size), interpolation=cv2.INTER_CUBIC))
    return out


class Extractor:
    """Wraps FaceLandmarker in VIDEO mode. Reuse one instance across clips."""

    def __init__(self, model_path: Path | None = None):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        path = ensure_model(model_path or MODEL_PATH)
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._vision = mp_vision
        self._opts = opts

    def run(self, frames: list[np.ndarray], fps: float, name: str = "",
            keep_frames: bool = False) -> VideoFeats:
        mp = self._mp
        T = len(frames)
        H, W = frames[0].shape[:2]
        scale_xyz = np.array([W, H, W], dtype=np.float32)
        ok = np.zeros(T, dtype=bool)
        bs = np.zeros((T, 52), dtype=np.float32)
        lm = np.zeros((T, 478, 3), dtype=np.float32)
        pose = np.zeros((T, 3), dtype=np.float32)
        bs_names: list[str] = []

        # A landmarker in VIDEO mode carries tracking state, so it must not be
        # shared between clips.
        with self._vision.FaceLandmarker.create_from_options(self._opts) as lmk:
            for t, frame in enumerate(frames):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts = int(round(t * 1000.0 / max(fps, 1e-3)))
                res = lmk.detect_for_video(image, ts)
                if not res.face_landmarks:
                    continue
                pts = res.face_landmarks[0]
                if len(pts) < 478:
                    continue
                lm[t] = np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float32) * scale_xyz
                if res.face_blendshapes:
                    cats = res.face_blendshapes[0]
                    if not bs_names:
                        bs_names = [c.category_name for c in cats]
                    bs[t] = np.array([c.score for c in cats], dtype=np.float32)
                if res.facial_transformation_matrixes:
                    m = np.array(res.facial_transformation_matrixes[0], dtype=np.float32)
                    pose[t] = _matrix_to_euler(m[:3, :3])
                ok[t] = True

        return VideoFeats(
            name=name or "clip",
            ok=ok, blendshapes=bs, landmarks=lm, pose=pose, bs_names=bs_names,
            width=W, height=H, fps=fps,
            frames_bgr=frames if keep_frames else None,
        )

    def locate_face(self, frame: np.ndarray, probe_size: int = 448,
                    levels: tuple[float, ...] = (1.0, 0.5, 0.28)) -> np.ndarray | None:
        """Coarse-to-fine tile search for a face too small for whole-frame detection.

        MediaPipe ships the *short-range* BlazeFace detector, which needs the face
        to fill a decent share of the input. A full-body shot puts the head at
        ~7% of frame height and detection simply returns nothing -- silently, as
        an all-invalid clip rather than an error. Scanning upscaled tiles gets the
        face back at the cost of a few detector calls on a single frame.

        Returns landmarks in the frame's own pixel coordinates, or None.
        """
        H, W = frame.shape[:2]
        # Portrait framing puts the head near the top-centre; searching there
        # first usually ends the scan after a handful of tiles.
        prior = np.array([W / 2.0, H / 3.0])
        for frac in levels:
            side = min(H, W) if frac >= 1.0 else int(round(min(H, W) * frac))
            step = max(1, side // 2)
            boxes = [(x, y, side, side)
                     for y in range(0, max(1, H - side + 1), step)
                     for x in range(0, max(1, W - side + 1), step)]
            boxes.sort(key=lambda b: np.hypot(b[0] + b[2] / 2 - prior[0],
                                              b[1] + b[3] / 2 - prior[1]))
            for box in boxes:
                chip = crop_frames([frame], box, probe_size)
                f = self.run(chip, 25.0)
                if f.n_valid:
                    s = box[2] / float(probe_size)
                    lm = f.landmarks[0].copy()
                    lm[:, 0] = lm[:, 0] * s + box[0]
                    lm[:, 1] = lm[:, 1] * s + box[1]
                    return lm
        return None

    def run_face_crop(self, frames: list[np.ndarray], fps: float, name: str = "",
                      crop_size: int = 512, margin: float = 1.9,
                      keep_frames: bool = False) -> VideoFeats:
        """Two-pass extraction on a fixed, upscaled face crop.

        Landmark precision scales with how many pixels the face occupies, and
        that precision is the granularity ceiling for the whole verifier: on
        480x360 studio footage the face is ~150 px and the measurement noise
        alone exceeds several frames' worth of real expression change. A single
        crop box (median over the clip, not per-frame) avoids introducing
        tracking jitter of its own.
        """
        probe = self.run(frames[::max(1, len(frames) // 24)], fps, name=name)
        if probe.n_valid:
            lm = probe.landmarks[probe.ok][:, :, :2]
        else:
            found = [self.locate_face(frames[t])
                     for t in np.linspace(0, len(frames) - 1, 3).astype(int)]
            found = [f for f in found if f is not None]
            if not found:
                return self.run(frames, fps, name=name, keep_frames=keep_frames)
            lm = np.stack(found)[:, :, :2]

        cx = float(np.median(lm[:, :, 0].mean(axis=1)))
        cy = float(np.median(lm[:, :, 1].mean(axis=1)))
        half = float(np.median(np.maximum(
            lm[:, :, 0].max(axis=1) - lm[:, :, 0].min(axis=1),
            lm[:, :, 1].max(axis=1) - lm[:, :, 1].min(axis=1)))) * margin / 2.0

        x0, y0 = int(round(cx - half)), int(round(cy - half))
        side = int(round(half * 2))
        box = (x0, y0, side, side)
        out = self.run(crop_frames(frames, box, crop_size), fps, name=name,
                       keep_frames=keep_frames)
        out.crop_box = box
        return out

    def from_path(self, path: str | Path, max_frames: int | None = None,
                  keep_frames: bool = False, resize_long: int | None = 512,
                  face_crop: bool = False, crop_size: int = 512) -> VideoFeats:
        frames, fps = read_video(path, max_frames=max_frames, resize_long=resize_long)
        if face_crop:
            return self.run_face_crop(frames, fps, name=Path(path).stem,
                                      crop_size=crop_size, keep_frames=keep_frames)
        return self.run(frames, fps, name=Path(path).stem, keep_frames=keep_frames)


def cache_path(video: str | Path, cache_dir: str | Path, tag: str = "") -> Path:
    return Path(cache_dir) / (Path(video).stem + tag + ".npz")


def save_feats(f: VideoFeats, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, name=f.name, ok=f.ok, blendshapes=f.blendshapes,
        landmarks=f.landmarks, pose=f.pose, bs_names=np.array(f.bs_names, dtype=object),
        width=f.width, height=f.height, fps=f.fps,
        crop_box=np.array(f.crop_box if f.crop_box else (-1, -1, -1, -1), dtype=np.int64),
    )


def load_feats(path: str | Path) -> VideoFeats:
    d = np.load(path, allow_pickle=True)
    cb = tuple(int(v) for v in d["crop_box"]) if "crop_box" in d.files else (-1, -1, -1, -1)
    return VideoFeats(
        name=str(d["name"]), ok=d["ok"], blendshapes=d["blendshapes"],
        landmarks=d["landmarks"], pose=d["pose"], bs_names=list(d["bs_names"]),
        width=int(d["width"]), height=int(d["height"]), fps=float(d["fps"]),
        crop_box=None if cb[2] <= 0 else cb,
    )


def landmarks_in_original(f: VideoFeats) -> np.ndarray:
    """Map landmarks from face-crop space back to the decoded frame's pixels.

    Everything geometric works in crop space, but ArcFace alignment and the
    background comparison consume the *original* frames; feeding them crop-space
    coordinates silently produces garbage rather than an error.
    """
    if f.crop_box is None:
        return f.landmarks
    x0, y0, side, _ = f.crop_box
    s = side / float(max(f.width, 1))
    out = f.landmarks.copy()
    out[..., 0] = out[..., 0] * s + x0
    out[..., 1] = out[..., 1] * s + y0
    out[..., 2] = out[..., 2] * s
    return out


def extract_cached(extractor: Extractor, video: str | Path, cache_dir: str | Path,
                   max_frames: int | None = None, face_crop: bool = True,
                   crop_size: int = 512) -> VideoFeats:
    cp = cache_path(video, cache_dir, tag=f"__c{crop_size}" if face_crop else "")
    if cp.exists():
        try:
            return load_feats(cp)
        except Exception:
            os.remove(cp)
    f = extractor.from_path(video, max_frames=max_frames, face_crop=face_crop,
                            crop_size=crop_size)
    save_feats(f, cp)
    return f
