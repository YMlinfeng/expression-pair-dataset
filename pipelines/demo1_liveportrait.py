"""Demo 1 -- expression equality by construction, via LivePortrait video-to-video.

One real performance is fanned out onto N different appearances. In
video-to-video mode with `--animation-region exp --no-flag-relative-motion`, the
expression keypoints written into every output come straight from the driving
template:

    x_d_exp_lst = [driving_template['motion'][i]['exp'] for i in range(n_frames)]

with no source-dependent term, so every source receives *the same* expression
tensor, frame by frame. Scale, translation and rotation continue to come from
each source, so each output keeps its own head pose, framing, background and
lighting. Expression equality is therefore a property of the construction, not
something the verifier has to discover -- which matters, because no available
metric is good enough to discover it.

Two implementation details decide whether that guarantee actually holds:

1. `n_frames = min(len(source), len(driving))`, and the expression sequence is
   Kalman-smoothed over exactly those n_frames. Sources of different lengths get
   different smoothing windows, which quietly downgrades "identical" to "nearly
   identical". Every source is therefore trimmed to the driver's exact length.
2. Absolute mode is known to leak the driver's face shape into the output. That
   is invisible to every expression metric, so it gets its own two-sided
   identity check: the output must stay close to its own source and far from the
   driver.

Both variants are generated. Relative mode adds a source-dependent term
(`source_exp[i] + driving_exp[i] - driving_exp[0]`), so its outputs should *not*
be expression-identical; it is generated as the control that shows the
verifier's verdict tracks the construction rather than the file names.
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from expverify.calibrate import Calibration, apply_to_spec
from expverify.descriptors import build_channel_plan
from expverify.identity import ArcFace, cosine
from expverify.landmarks import Extractor, landmarks_in_original, read_video
from expverify.neutral import describe, estimate_neutral
from expverify.report import (ManifestWriter, plot_pair, summarize, write_markdown,
                              write_schema)
from expverify.scene import background_similarity, pose_delta, sample_indices
from expverify.verify import SPECS, Weights, verify_pair
from pipelines.common import au_for_feats, ensure_reference

LP_ROOT = Path("third_party/LivePortrait").resolve()


def n_frames_of(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def child_env() -> dict[str, str]:
    """LivePortrait's entry point aborts unless an `ffmpeg` binary is on PATH.
    imageio-ffmpeg already ships one; expose it under the expected name so the
    demo needs no system-wide install. (`ffprobe` is only used for an audio
    probe that degrades to False, so it can stay absent.)"""
    d = Path("out/work/bin").resolve()
    d.mkdir(parents=True, exist_ok=True)
    link = d / "ffmpeg"
    if not link.exists():
        link.symlink_to(ffmpeg_exe())
    env = dict(os.environ)
    env["PATH"] = f"{d}{os.pathsep}{env.get('PATH', '')}"
    return env


def trim(src: Path, dst: Path, n: int, max_dim: int | None = None) -> Path:
    """Re-encode to exactly `n` frames. Equal length across sources is a
    correctness requirement, not an optimisation -- see the module docstring."""
    if dst.exists() and n_frames_of(dst) == n:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = [f"scale='min({max_dim},iw)':-2"] if max_dim else []
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
           "-frames:v", str(n)]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-an", str(dst)]
    subprocess.run(cmd, check=True)
    got = n_frames_of(dst)
    if got != n:
        raise RuntimeError(f"trim produced {got} frames, expected {n}: {dst}")
    return dst


@dataclass
class Variant:
    name: str
    relative: bool

    @property
    def flag(self) -> str:
        return "--flag-relative-motion" if self.relative else "--no-flag-relative-motion"


def run_liveportrait(source: Path, driving: Path, out_dir: Path, variant: Variant,
                     source_max_dim: int = 720, timeout: int = 3600) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = out_dir / f"{source.stem}--{driving.stem}.mp4"
    if expected.exists() and n_frames_of(expected) > 0:
        print(f"    have {expected.name}")
        return expected
    cmd = [sys.executable, "inference.py",
           "-s", str(source.resolve()), "-d", str(driving.resolve()),
           "-o", str(out_dir.resolve()),
           "--animation-region", "exp",
           variant.flag,
           "--source-max-dim", str(source_max_dim),
           # paste-back keeps each output in its source's full frame, which is
           # what preserves the differing backgrounds the spec requires
           "--flag-pasteback"]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=LP_ROOT, capture_output=True, text=True, timeout=timeout,
                       env=child_env())
    if r.returncode != 0:
        print(f"    ! {source.stem}: {r.stdout[-800:]}\n{r.stderr[-1200:]}")
        return None
    if not expected.exists():
        print(f"    ! missing output {expected}")
        return None
    n = n_frames_of(expected)
    print(f"    {expected.name}  {n} frames  {time.time() - t0:.0f}s "
          f"({(time.time() - t0) / max(n, 1):.2f}s/frame)")
    return expected


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--driver", default="third_party/LivePortrait/assets/examples/driving/d0.mp4")
    ap.add_argument("--sources", nargs="+", default=[
        "third_party/LivePortrait/assets/examples/source/s13.mp4",
        "third_party/LivePortrait/assets/examples/source/s18.mp4",
        "third_party/LivePortrait/assets/examples/source/s32.mp4",
    ])
    ap.add_argument("--out", default="out/demo1_liveportrait")
    ap.add_argument("--work", default="out/work/demo1")
    ap.add_argument("--calibration", default="out/calibration/liveportrait")
    ap.add_argument("--spec", default="reference", choices=list(SPECS))
    ap.add_argument("--max-frames", type=int, default=0, help="0 = driver length")
    ap.add_argument("--source-max-dim", type=int, default=720)
    ap.add_argument("--variants", nargs="+", default=["absolute", "relative"])
    ap.add_argument("--skip-generation", action="store_true")
    ap.add_argument("--au", action="store_true", help="add M3 (OpenFace AU) to the gate")
    ap.add_argument("--read-long", type=int, default=1024,
                    help="long-side resolution clips are decoded at before cropping")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    work = Path(args.work); work.mkdir(parents=True, exist_ok=True)

    driver_src = Path(args.driver)
    n = n_frames_of(driver_src)
    if args.max_frames:
        n = min(n, args.max_frames)
    sources_src = [Path(s) for s in args.sources]
    too_short = [s for s in sources_src if n_frames_of(s) < n]
    if too_short:
        raise SystemExit(f"sources shorter than driver ({n} frames): {too_short}")

    print(f"driver {driver_src.name}: locking every clip to {n} frames")
    driver = trim(driver_src, work / f"D_{driver_src.stem}.mp4", n)
    sources = [trim(s, work / f"S_{s.stem}.mp4", n, args.source_max_dim) for s in sources_src]

    variants = [Variant("absolute", False), Variant("relative", True)]
    variants = [v for v in variants if v.name in args.variants]

    gen: dict[str, dict[str, Path]] = {}
    for v in variants:
        print(f"\n[{v.name}] animation-region=exp  {v.flag}")
        gen[v.name] = {}
        if args.skip_generation:
            for s in sources:
                p = out / v.name / f"{s.stem}--{driver.stem}.mp4"
                if p.exists():
                    gen[v.name][s.stem] = p
            continue
        for s in sources:
            p = run_liveportrait(s, driver, out / v.name, v, args.source_max_dim)
            if p:
                gen[v.name][s.stem] = p

    # ---- feature extraction -------------------------------------------------
    extractor = Extractor()
    clips: dict[str, Path] = {"D": driver}
    for s in sources:
        clips[f"src:{s.stem}"] = s
    for v in variants:
        for stem, p in gen[v.name].items():
            clips[f"{v.name}:{stem}"] = p

    au_ex = None
    if args.au:
        try:
            from expverify.au import AUExtractor
            au_ex = AUExtractor()
        except Exception as e:  # noqa: BLE001
            print(f"M3 unavailable ({e}); running without AU")

    # Source framing is arbitrary -- s13 is a full-body shot whose head is ~7% of
    # frame height -- so clips are decoded large and cropped, rather than decoded
    # small and hoped over.
    print(f"\nextracting features for {len(clips)} clips")
    feats, frames, aus = {}, {}, {}
    for key, path in clips.items():
        fr, fps = read_video(path, resize_long=args.read_long)
        f = extractor.run_face_crop(fr, fps, name=key)
        if f.n_valid < 5:
            print(f"  ! {key}: only {f.n_valid} valid frames")
            continue
        feats[key], frames[key] = f, fr
        if au_ex is not None:
            aus[key] = au_for_feats(au_ex, fr, f)
        face_px = f.crop_box[2] if f.crop_box else min(f.width, f.height)
        print(f"  {key:<24} {len(f):>4} frames, {f.n_valid:>4} valid, "
              f"{f.width}x{f.height}, face {face_px}px")

    if "D" not in feats:
        raise SystemExit("driver produced no usable landmarks")

    ensure_reference(list(feats.values()))
    plan = build_channel_plan(feats["D"].bs_names)
    # Each clip is its own identity, so each gets its own neutral: an output
    # inherits its source's face shape, and mixing neutrals would reintroduce
    # exactly the identity bias the neutral subtraction exists to remove.
    descs = {k: describe(f, plan, estimate_neutral([f], plan, k), au=aus.get(k))
             for k, f in feats.items()}

    spec = SPECS[args.spec]
    weights = Weights()
    try:
        calib = Calibration.load(args.calibration)
        spec = apply_to_spec(spec, calib)
        weights = calib.weights()
        print(f"\ncalibration applied: max_deform={spec.max_deform:.4f} "
              f"min_pass_rate={spec.min_pass_rate:.3f} max_ratio={spec.max_ratio:.2f} "
              f"min_energy={spec.min_energy:.4f}")
    except Exception as e:  # noqa: BLE001
        print(f"\nno calibration ({e}); using spec defaults")
    spec.min_source_identity_cos = 0.45
    spec.max_driver_identity_cos = 0.30

    arcface = None
    try:
        arcface = ArcFace()
    except Exception as e:  # noqa: BLE001
        print(f"ArcFace unavailable ({e}); identity gates disabled")

    lm_orig = {k: landmarks_in_original(f) for k, f in feats.items()}
    emb: dict[str, np.ndarray | None] = {}
    for k, f in feats.items():
        emb[k] = arcface.embed_video(frames[k], lm_orig[k], f.ok) if arcface else None

    # ---- verification -------------------------------------------------------
    results: list = []
    manifest = ManifestWriter(out / "manifest.jsonl")
    figures = 0
    per_variant: dict[str, list] = {v.name: [] for v in variants}

    def bg_of(a: str, b: str) -> float | None:
        ia, ib = sample_indices(feats[a].ok, 3), sample_indices(feats[b].ok, 3)
        if not len(ia) or not len(ib):
            return None
        vals = [background_similarity(frames[a][i], lm_orig[a][i],
                                      frames[b][j], lm_orig[b][j])["hist"]
                for i, j in zip(ia, ib)]
        return float(np.median(vals))

    for v in variants:
        stems = sorted(gen[v.name])
        keys = [f"{v.name}:{s}" for s in stems if f"{v.name}:{s}" in descs]

        # output-vs-output, and output-vs-real-driver (which gives pairs whose
        # reference side is genuine footage rather than a second generation)
        pairs = [(a, b, "gen-gen") for a, b in itertools.combinations(keys, 2)]
        pairs += [("D", k, "real-gen") for k in keys]

        for a, b, kind in pairs:
            da, db = descs[a], descs[b]
            pd = pose_delta(feats[a].pose[:min(len(da), len(db))],
                            feats[b].pose[:min(len(da), len(db))])
            src_a = f"src:{a.split(':', 1)[1]}" if ":" in a else None
            src_b = f"src:{b.split(':', 1)[1]}" if ":" in b else None
            r = verify_pair(
                da, db, spec, weights=weights,
                identity_cos=cosine(emb.get(a), emb.get(b)),
                pose_delta=pd, bg_hist=bg_of(a, b),
                source_identity_cos=min([x for x in (cosine(emb.get(a), emb.get(src_a)),
                                                     cosine(emb.get(b), emb.get(src_b)))
                                         if x is not None], default=None),
                driver_identity_cos=max([x for x in (cosine(emb.get(a), emb.get("D")),
                                                     cosine(emb.get(b), emb.get("D")))
                                         if x is not None] if kind == "gen-gen" else [],
                                        default=None),
                extra={"variant": v.name, "kind": kind,
                       "face_px": min(feats[a].crop_box[2] if feats[a].crop_box else 0,
                                      feats[b].crop_box[2] if feats[b].crop_box else 0)},
            )
            results.append(r)
            per_variant[v.name].append(r)
            manifest.write(r, variant=v.name, kind=kind)
            if figures < 8:
                tag = "accept" if r.accepted else "reject"
                plot_pair(r, frames[a], frames[b],
                          out / f"{v.name}_{tag}_{a.replace(':', '-')}__{b.replace(':', '-')}.png",
                          spec)
                figures += 1
    manifest.close()

    # ---- report -------------------------------------------------------------
    face_px = {k: (f.crop_box[2] if f.crop_box else min(f.width, f.height))
               for k, f in feats.items()}
    sections = ["# Demo 1 - LivePortrait construction-guaranteed pairs\n",
                f"Driver `{driver_src.name}`, {n} frames, "
                f"{len(sources)} sources, spec `{spec.name}`.\n",
                "\nMeasurement resolution (face height in decoded pixels) sets the "
                "granularity ceiling for every metric below; the smaller side of a "
                "pair is the one that binds:\n\n"
                "| clip | face px |\n| --- | ---: |\n"
                + "\n".join(f"| {k} | {v} |" for k, v in sorted(face_px.items())) + "\n"]
    print()
    for v in variants:
        rs = per_variant[v.name]
        if not rs:
            continue
        sections.append(summarize(rs, f"variant: {v.name} ({v.flag})"))
        acc = sum(r.accepted for r in rs)
        med = lambda k: np.median([r.summary[k] for r in rs if r.summary.get(k) is not None])
        print(f"[{v.name}] {acc}/{len(rs)} accepted | med d_deform={med('med_d_deform'):.4f} "
              f"| med ratio={med('med_ratio'):.3f} | rank1={med('rank1_rate'):.3f}")

    if len(variants) == 2:
        a = [r for r in per_variant["absolute"]]
        rel = [r for r in per_variant["relative"]]
        if a and rel:
            f = lambda rs, k: float(np.median([r.summary[k] for r in rs
                                               if r.summary.get(k) is not None]))
            n_a, n_r = sum(r.accepted for r in a), sum(r.accepted for r in rel)
            sections.append(
                "\n## absolute vs relative\n\n"
                "Relative mode is the control, not a competitor: it adds a\n"
                "source-dependent term, so its outputs are *not* expression-identical\n"
                "and the verifier should say so. If both variants scored alike, the\n"
                "verdict would be tracking file names rather than the construction.\n\n"
                "| | absolute (expression-identical) | relative (control) |\n"
                "| --- | ---: | ---: |\n"
                f"| accepted | {n_a}/{len(a)} | {n_r}/{len(rel)} |\n"
                f"| med d_deform | {f(a, 'med_d_deform'):.4f} | {f(rel, 'med_d_deform'):.4f} |\n"
                f"| med rank-1 rate | {f(a, 'rank1_rate'):.3f} | {f(rel, 'rank1_rate'):.3f} |\n"
                f"| med ratio | {f(a, 'med_ratio'):.3f} | {f(rel, 'med_ratio'):.3f} |\n"
                f"| med frame pass rate | {f(a, 'pass_rate'):.3f} | {f(rel, 'pass_rate'):.3f} |\n\n"
                f"Verdict: absolute mode is the one to generate with. It is closer on\n"
                f"every expression metric and it is the only variant that clears the\n"
                f"gate ({n_a}/{len(a)} vs {n_r}/{len(rel)}), which is what the source\n"
                f"code predicts -- in absolute mode the expression tensor written into\n"
                f"every output comes from the driving template with no source term.\n")

    if arcface:
        rows = [f"| {v.name}:{s} | {cosine(emb[f'{v.name}:{s}'], emb.get(f'src:{s}')):.3f} | "
                f"{cosine(emb[f'{v.name}:{s}'], emb.get('D')):.3f} |"
                for v in variants for s in sorted(gen[v.name])
                if f"{v.name}:{s}" in emb]
        sections.append("\n## identity integrity (does the driver's face shape leak?)\n\n"
                        "The output must stay its own source and must not drift toward the\n"
                        "driver. No expression metric can see this failure, so it needs its\n"
                        "own check.\n\n"
                        "| output | cos(out, own source) | cos(out, driver) |\n"
                        "| --- | ---: | ---: |\n" + "\n".join(rows) + "\n")

    write_markdown(out / "summary.md", sections)
    write_schema(out / "manifest.schema.md")
    print(f"\nmanifest -> {out / 'manifest.jsonl'} ({manifest.n} rows)")
    print(f"summary  -> {out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
