"""Appearance-only augmentation: changes how a clip looks, never its geometry.

Two uses.

1. Calibration. Running a clip and its augmented twin through the whole verifier
   gives *construction-guaranteed positives* whose expression is identical by
   definition, so the resulting distances measure the verifier's own noise
   floor. Any acceptance threshold below that floor is unachievable; any
   threshold above the hard-negative distribution is meaningless.

2. Style/background variety in generated pairs. Deliberately not a diffusion
   img2img pass: that has no expression-preservation constraint and no temporal
   model, so it perturbs mouth shape and eye state frame to frame -- destructive
   for a dataset whose entire premise is expression equality. Colour grading,
   grain and geometric jitter touch no face geometry at all, and the similarity
   jitter is provably removed by the canonical frame in descriptors.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AugmentConfig:
    brightness: float = 0.0     # additive, in [-1, 1] scaled to 255
    contrast: float = 1.0
    saturation: float = 1.0
    gamma: float = 1.0
    color_shift: tuple[float, float, float] = (0.0, 0.0, 0.0)   # BGR additive
    noise_sigma: float = 0.0
    blur_sigma: float = 0.0
    jpeg_quality: int = 0       # 0 disables
    rotate_deg: float = 0.0
    scale: float = 1.0
    translate: tuple[float, float] = (0.0, 0.0)   # fraction of width/height
    downscale: float = 1.0      # resolution round-trip factor, <1 degrades


# Tiered so the noise floor can be measured at the strength actually shipped,
# rather than at a worst case. Heavy presets degrade the landmark detector
# enough to swamp several frames' worth of real expression change, which makes
# them a stress test, not a calibration reference.
TIERS: dict[str, dict[str, AugmentConfig]] = {
    # Colour only: no geometry, no resampling, near-lossless. This is the tier
    # we actually use for style variety in shipped pairs.
    "mild": {
        "warm": AugmentConfig(brightness=0.05, contrast=1.10, saturation=1.20,
                              gamma=0.95, color_shift=(-12.0, -3.0, 15.0),
                              jpeg_quality=95),
        "cool": AugmentConfig(brightness=-0.035, contrast=0.94, saturation=0.85,
                              gamma=1.08, color_shift=(16.0, 3.0, -10.0),
                              jpeg_quality=95),
        "flat": AugmentConfig(contrast=0.88, saturation=0.72, gamma=1.12,
                              color_shift=(6.0, 2.0, 4.0), jpeg_quality=95),
    },
    "medium": {
        "warm_grain": AugmentConfig(brightness=0.06, contrast=1.12, saturation=1.25,
                                    gamma=0.95, color_shift=(-14.0, -4.0, 18.0),
                                    noise_sigma=3.0, jpeg_quality=85, downscale=0.9),
        "cool_soft": AugmentConfig(brightness=-0.04, contrast=0.92, saturation=0.8,
                                   gamma=1.1, color_shift=(20.0, 4.0, -12.0),
                                   blur_sigma=0.6, jpeg_quality=85),
        "jitter": AugmentConfig(contrast=1.15, saturation=1.05, gamma=0.95,
                                noise_sigma=2.0, rotate_deg=3.0, scale=0.95,
                                translate=(0.015, -0.01), jpeg_quality=88),
    },
    "heavy": {
        "warm_grain": AugmentConfig(brightness=0.06, contrast=1.12, saturation=1.25,
                                    gamma=0.95, color_shift=(-14.0, -4.0, 18.0),
                                    noise_sigma=7.0, jpeg_quality=62, downscale=0.75),
        "cool_soft": AugmentConfig(brightness=-0.04, contrast=0.92, saturation=0.8,
                                   gamma=1.1, color_shift=(20.0, 4.0, -12.0),
                                   blur_sigma=1.1, jpeg_quality=80),
        "hi_contrast_jitter": AugmentConfig(contrast=1.3, saturation=1.1, gamma=0.88,
                                            noise_sigma=4.0, rotate_deg=4.0, scale=0.93,
                                            translate=(0.02, -0.015)),
    },
}

PRESETS: dict[str, AugmentConfig] = {f"{tier}:{k}": v
                                     for tier, d in TIERS.items() for k, v in d.items()}


def _apply_photometric(img: np.ndarray, c: AugmentConfig, rng: np.random.Generator) -> np.ndarray:
    x = img.astype(np.float32)
    if c.saturation != 1.0:
        hsv = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * c.saturation, 0, 255)
        x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    x = (x - 128.0) * c.contrast + 128.0 + c.brightness * 255.0
    x = x + np.array(c.color_shift, dtype=np.float32)
    if c.gamma != 1.0:
        x = 255.0 * np.power(np.clip(x, 0, 255) / 255.0, c.gamma)
    if c.noise_sigma > 0:
        x = x + rng.normal(0.0, c.noise_sigma, size=x.shape).astype(np.float32)
    x = np.clip(x, 0, 255).astype(np.uint8)
    if c.blur_sigma > 0:
        x = cv2.GaussianBlur(x, (0, 0), c.blur_sigma)
    if c.downscale < 1.0:
        h, w = x.shape[:2]
        small = cv2.resize(x, (max(8, int(w * c.downscale)), max(8, int(h * c.downscale))),
                           interpolation=cv2.INTER_AREA)
        x = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    if c.jpeg_quality:
        ok, buf = cv2.imencode(".jpg", x, [int(cv2.IMWRITE_JPEG_QUALITY), int(c.jpeg_quality)])
        if ok:
            x = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return x


def _apply_geometric(img: np.ndarray, c: AugmentConfig) -> np.ndarray:
    if c.rotate_deg == 0.0 and c.scale == 1.0 and c.translate == (0.0, 0.0):
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), c.rotate_deg, c.scale)
    M[0, 2] += c.translate[0] * w
    M[1, 2] += c.translate[1] * h
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def augment_frames(frames: list[np.ndarray], preset: str | AugmentConfig,
                   seed: int = 0) -> list[np.ndarray]:
    c = PRESETS[preset] if isinstance(preset, str) else preset
    rng = np.random.default_rng(seed)
    return [_apply_photometric(_apply_geometric(f, c), c, rng) for f in frames]


def tier_presets(tier: str) -> list[str]:
    return [f"{tier}:{k}" for k in TIERS[tier]]
