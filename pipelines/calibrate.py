"""Fit verifier thresholds against construction-guaranteed positives and
temporally-adjacent hard negatives.

Two positive sources, and the difference between them is the whole story:

- `--positives augment` uses appearance-augmented twins of real clips: same
  face, same geometry, different look. That measures the verifier's *noise
  floor* -- how far apart two measurements of one identical face can land.
- `--positives liveportrait` uses Demo 1's outputs: different people driven by
  one expression tensor, frame for frame. That measures the *cross-identity
  floor* -- how far apart two different faces land when their expression really
  is identical, which is unavoidably larger because two faces do not render the
  same expression into the same geometry.

Thresholds fitted on the first are unreachable by anything cross-identity, so
the second is what the production gate should be calibrated on. Run both: the
gap between them is the price of crossing identities, and it is worth knowing.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from expverify.augment import augment_frames, tier_presets
from expverify.calibrate import (apply_to_spec, calibrate, calibrate_pair_level,
                                 collect_positives)
from expverify.landmarks import Extractor, read_video
from expverify.neutral import describe, estimate_neutral
from expverify.report import calibration_markdown, write_markdown
from expverify.verify import SPECS, Weights
from pipelines.common import au_for_feats, build_corpus, load_cremad_clips


def make_augmented_positives(corpus, clips, extractor, n_clips: int, tier: str,
                             seed: int = 0, au_ex=None):
    """(clip, augmented twin) pairs; each side gets its own neutral estimate."""
    rng = np.random.default_rng(seed)
    chosen = list(rng.choice(len(clips), size=min(n_clips, len(clips)), replace=False))
    presets = tier_presets(tier)
    pairs = []
    for n, ci in enumerate(tqdm(chosen, desc=f"augment[{tier}]", ncols=88)):
        c = clips[int(ci)]
        if c.stem not in corpus.descs:
            continue
        try:
            frames, fps = read_video(c.path)
            aug = augment_frames(frames, presets[n % len(presets)], seed=n)
            f_aug = extractor.run_face_crop(aug, fps, name=f"{c.stem}__aug_{tier}")
            if f_aug.n_valid < 5:
                continue
            neutral = estimate_neutral([f_aug], corpus.plan, f"{c.person}__aug")
            au = au_for_feats(au_ex, aug, f_aug) if au_ex is not None else None
            d_aug = describe(f_aug, corpus.plan, neutral, au=au)
        except Exception as e:  # noqa: BLE001
            print(f"  ! augment {c.stem}: {e}")
            continue
        pairs.append((corpus.descs[c.stem], d_aug))
    return pairs


def make_liveportrait_positives(lp_dir: str, work_dir: str, variant: str, plan,
                                extractor, au_ex=None, max_clips: int = 8):
    """Cross-identity positives whose expression latent is identical by construction.

    Augmented twins measure the verifier's noise floor on *one* face; they say
    nothing about how much of the distance between two different faces is
    unavoidable rendering difference rather than expression disagreement.
    Thresholds fitted on twins are therefore far too tight for any real
    cross-identity pair -- which is exactly what the first Demo 1 run showed.
    These pairs are the right calibration set: different people, same expression
    tensor, frame for frame.
    """
    lp = Path(lp_dir) / variant
    outs = sorted(p for p in lp.glob("*.mp4") if not p.stem.endswith("_concat"))
    driver = next(iter(sorted(Path(work_dir).glob("D_*.mp4"))), None)
    if not outs:
        raise FileNotFoundError(f"no generated clips in {lp}")

    descs = {}
    for p in ([driver] if driver else []) + outs[:max_clips]:
        frames, fps = read_video(p, resize_long=1024)
        f = extractor.run_face_crop(frames, fps, name=p.stem)
        if f.n_valid < 8:
            print(f"  ! {p.stem}: {f.n_valid} valid frames")
            continue
        au = au_for_feats(au_ex, frames, f) if au_ex is not None else None
        descs[p.stem] = describe(f, plan, estimate_neutral([f], plan, p.stem), au=au)

    gen = [k for k in descs if k != (driver.stem if driver else None)]
    pairs = [(descs[a], descs[b]) for a, b in itertools.combinations(sorted(gen), 2)]
    if driver and driver.stem in descs:
        pairs += [(descs[driver.stem], descs[k]) for k in sorted(gen)]
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/cremad")
    ap.add_argument("--cache", default="out/cache/cremad")
    ap.add_argument("--out", default="out/calibration/liveportrait")
    ap.add_argument("--n-pos-clips", type=int, default=24)
    ap.add_argument("--n-corpus-clips", type=int, default=160)
    ap.add_argument("--precision", type=float, default=0.95)
    ap.add_argument("--hard-offset", type=int, default=3)
    ap.add_argument("--tier", default="mild", choices=["mild", "medium", "heavy"],
                    help="augmentation tier used as the positive set")
    ap.add_argument("--all-tiers", action="store_true",
                    help="also report the noise floor of the other tiers")
    ap.add_argument("--spec", default="editing", choices=list(SPECS),
                    help="spec whose frame gate is used for pair-level fitting")
    ap.add_argument("--energy-percentile", type=float, default=0.35)
    ap.add_argument("--au", action="store_true",
                    help="add M3 (OpenFace AU activations) to the gate")
    ap.add_argument("--positives", default="augment", choices=["augment", "liveportrait"],
                    help="augment = same face, different look (noise floor); "
                         "liveportrait = different faces, identical expression latent")
    ap.add_argument("--lp-dir", default="out/demo1_liveportrait")
    ap.add_argument("--lp-work", default="out/work/demo1")
    ap.add_argument("--lp-variant", default="absolute")
    args = ap.parse_args()

    clips = load_cremad_clips(args.data)
    rng = np.random.default_rng(0)
    if len(clips) > args.n_corpus_clips:
        idx = rng.choice(len(clips), size=args.n_corpus_clips, replace=False)
        clips = [clips[int(i)] for i in sorted(idx)]
    print(f"corpus: {len(clips)} clips, {len({c.person for c in clips})} people")

    au_ex = None
    if args.au:
        try:
            from expverify.au import AUExtractor
            au_ex = AUExtractor()
        except Exception as e:  # noqa: BLE001
            print(f"M3 unavailable ({e}); calibrating without AU")

    extractor = Extractor()
    corpus = build_corpus(clips, args.cache, extractor, au_extractor=au_ex)
    print(f"descriptors: {len(corpus.descs)}  M1 dim={corpus.plan.dim}"
          f"{'  +M3 AU' if au_ex else ''}")

    if args.positives == "liveportrait":
        pos_pairs = make_liveportrait_positives(args.lp_dir, args.lp_work, args.lp_variant,
                                                corpus.plan, extractor, au_ex=au_ex)
        pos_label = f"liveportrait:{args.lp_variant}"
        print(f"positive pairs (cross-identity, expression-identical by "
              f"construction, {args.lp_variant}): {len(pos_pairs)}")
    else:
        pos_pairs = make_augmented_positives(corpus, clips, extractor,
                                             args.n_pos_clips, args.tier, au_ex=au_ex)
        pos_label = f"augmented_twin:{args.tier}"
        print(f"positive pairs (appearance-augmented twins, tier={args.tier}): "
              f"{len(pos_pairs)}")

    calib = calibrate(pos_pairs, corpus.desc_list(),
                      offsets=(3, 5, 10), precision=args.precision,
                      hard_offset=args.hard_offset,
                      energy_percentile=args.energy_percentile)
    calib.report["positive_source"] = pos_label

    # Second stage: with the per-metric thresholds installed, fit the pair-level
    # pass rate by running the real verifier on aligned vs. shifted positives.
    spec = apply_to_spec(SPECS[args.spec], calib)
    pl = calibrate_pair_level(spec, pos_pairs, calib.weights(), precision=args.precision)
    if pl.get("min_pass_rate") is not None:
        calib.thresholds["min_pass_rate"] = pl["min_pass_rate"]
    calib.report["pair_level"] = pl

    if args.all_tiers and args.positives == "augment":
        w = Weights(bs=calib.bs_sigma).bs_weights(corpus.plan.dim)
        floors = {}
        for tier in ("mild", "medium", "heavy"):
            pp = (pos_pairs if tier == args.tier else
                  make_augmented_positives(corpus, clips, extractor,
                                           max(8, args.n_pos_clips // 2), tier))
            d = collect_positives(pp, w)
            floors[tier] = {m: float(np.median(v)) for m, v in d.items()}
        calib.report["noise_floor_by_tier"] = floors
        print("\nnoise floor (median distance) by augmentation tier")
        print(f"{'tier':<8} " + " ".join(f"{m:>10}" for m in ("d_bs", "d_deform",
                                                              "d_gaze", "d_region")))
        for tier, d in floors.items():
            print(f"{tier:<8} " + " ".join(f"{d[m]:>10.4f}" for m in ("d_bs", "d_deform",
                                                                      "d_gaze", "d_region")))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    calib.save(args.out)
    write_markdown(Path(args.out).with_suffix(".md"),
                   [calibration_markdown(calib.thresholds, calib.report)])

    print(f"\ncalibration vs. paired hard negatives at +{args.hard_offset} frames"
          f" @ precision>={args.precision}")
    hdr = f"{'metric':<10} {'thr':>8} {'pos_med':>8} {'hard_med':>9} {'null_med':>9} " \
          f"{'AUC':>6} {'recall':>7} {'sep':>5}"
    print(hdr)
    print("-" * len(hdr))
    for m, info in calib.report["metrics"].items():
        print(f"{m:<10} {info['threshold']:>8.4f} {info['pos_median']:>8.4f} "
              f"{info['hardneg_median']:>9.4f} {info['null_median']:>9.4f} "
              f"{info['auc']:>6.3f} {info['recall']:>7.3f} "
              f"{'yes' if info['separated'] else 'NO':>5}")
        if info.get("target_precision_unreachable"):
            print("           ^ target precision unreachable; fell back to pos q90")

    print("\nAUC vs. temporal offset of the negative (how fine a distinction each metric makes)")
    offs = sorted(calib.report["metrics"]["d_deform"]["auc_by_offset"], key=int)
    print(f"{'metric':<10} " + " ".join(f"{'+' + o + 'f':>8}" for o in offs))
    for m, info in calib.report["metrics"].items():
        print(f"{m:<10} " + " ".join(f"{info['auc_by_offset'][o]:>8.3f}" for o in offs))

    red = calib.report.get("redundancy") or {}
    if red:
        print("\nmetric redundancy (Pearson r between per-frame distances)")
        print("  a conjunctive gate over two near-collinear metrics is one gate twice")
        for k, v in sorted(red.items(), key=lambda kv: -abs(kv[1])):
            print(f"  {k:<22} {v:+.3f}{'   <- collinear' if abs(v) > 0.9 else ''}")

    r1 = calib.report["rank1"]
    print("\nrank-1 temporal identifiability on construction-guaranteed positives")
    print("  (rates on the testable subset -- frames whose own +/-k neighbours differ)")
    print(f"{'case':<10} {'n_test':>7} {'exact':>7} {'+/-1':>7} {'+/-2':>7} "
          f"{'|off|med':>9} {'mean off':>9} {'testable':>9}")
    for case, d in r1.items():
        if not d.get("n_testable"):
            continue
        print(f"{case:<10} {d['n_testable']:>7} {d['exact_rate_testable']:>7.3f} "
              f"{d['within_tol_rate_testable']:>7.3f} {d['within_2_rate_testable']:>7.3f} "
              f"{d['median_abs_offset_testable']:>9.1f} {d['mean_offset']:>9.2f} "
              f"{(d['testable_rate'] or 0):>9.3f}")
    print("  shift+/-5 are negative controls: the argmin must follow the shift,")
    print("  otherwise the ranking test is measuring nothing.")

    rr = calib.report.get("ratio")
    if rr:
        print(f"\nrank-1 ratio gate  max_ratio={rr['threshold']:.3f}  "
              f"(aligned median {rr['pos_median']:.3f}, shifted median {rr['neg_median']:.3f}, "
              f"AUC={rr['auc']:.3f})")
    en = calib.report.get("energy")
    if en:
        print(f"expressiveness gate  min_energy={en['min_energy']:.4f} "
              f"(corpus p{int(en['percentile'] * 100)})")

    pl = calib.report.get("pair_level") or {}
    if pl.get("gate_pass_rates"):
        print("\nper-gate pass rate on construction-guaranteed positives")
        print("  (the lowest row is what limits the whole conjunctive gate)")
        for k, v in sorted(pl["gate_pass_rates"].items(), key=lambda kv: kv[1]):
            print(f"  {k:<10} {v:.3f}")

    if pl.get("min_pass_rate") is not None:
        print("\npair-level pass rate through the full conjunctive frame gate")
        print(f"  aligned pairs   n={pl['aligned_n']:<4} median={pl['aligned_median']:.3f} "
              f"q05={pl['aligned_q05']:.3f}")
        print(f"  shifted pairs   n={pl['shifted_n']:<4} median={pl['shifted_median']:.3f} "
              f"q95={pl['shifted_q95']:.3f}")
        print(f"  AUC={pl['auc']:.3f}  fitted min_pass_rate={pl['min_pass_rate']:.3f}  "
              f"recall={pl['recall']:.3f}  precision={pl['precision']:.3f}  "
              f"fully separated={'yes' if pl['separated'] else 'no'}")

    print(f"\nsaved -> {args.out}.npz / .json / .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
