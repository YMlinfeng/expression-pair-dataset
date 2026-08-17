"""Fetch only the LivePortrait weights the human pipeline needs (~660 MB).

The HF repo also carries the animal models (~1.4 GB) and the full buffalo_l
pack; neither is used by video-to-video human retargeting, so both are skipped.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/KwaiVGI/LivePortrait/resolve/main"
FILES = [
    "liveportrait/base_models/appearance_feature_extractor.pth",
    "liveportrait/base_models/motion_extractor.pth",
    "liveportrait/base_models/spade_generator.pth",
    "liveportrait/base_models/warping_module.pth",
    "liveportrait/retargeting_models/stitching_retargeting_module.pth",
    "liveportrait/landmark.onnx",
    "insightface/models/buffalo_l/2d106det.onnx",
    "insightface/models/buffalo_l/det_10g.onnx",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="third_party/LivePortrait/pretrained_weights")
    args = ap.parse_args()
    root = Path(args.out)

    total = 0
    for rel in FILES:
        dst = root / rel
        if dst.exists() and dst.stat().st_size > 1000:
            total += dst.stat().st_size
            print(f"  have {rel}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"  get  {rel}")
        urllib.request.urlretrieve(f"{BASE}/{rel}", dst)
        total += dst.stat().st_size
    print(f"total {total / 1e6:.0f} MB in {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
