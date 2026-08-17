"""Demo 2 -- real cross-identity pairs from CREMA-D, with no generative model.

91 actors speak the same 12 sentences in 6 emotions at 4 intensities, so a
(sentence, emotion, intensity) group is a set of different people performing the
same prescribed thing. Audio DTW puts corresponding moments on the same index,
then the verifier decides.

The expected outcome is a low acceptance rate, and that is the finding, not a
failure: "same sentence, same emotion label, same intensity" is a *coarse*
constraint, and two actors given it will still produce visibly different faces.
This demo exists to show what real cross-actor data can and cannot deliver, and
to supply hard negatives -- same emotion, different intensity -- whose labels
are human-assigned and therefore independent of every metric being calibrated.

Pairs are scored against the `editing` spec: CREMA-D is frontal, single-backdrop
studio footage, so demanding pose and background dissimilarity would reject
every pair for a reason that has nothing to do with expression.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from expverify.audio import align_videos
from expverify.calibrate import Calibration, apply_to_spec
from expverify.identity import ArcFace, cosine, identity_separation
from expverify.landmarks import Extractor, landmarks_in_original, read_video
from expverify.report import (ManifestWriter, plot_pair, summarize, write_markdown,
                              write_schema)
from expverify.scene import pose_delta
from expverify.verify import SPECS, Weights, verify_pair
from pipelines.common import build_corpus, load_cremad_clips


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/cremad")
    ap.add_argument("--cache", default="out/cache/cremad")
    ap.add_argument("--out", default="out/demo2_cremad")
    ap.add_argument("--calibration", default="out/calibration/liveportrait")
    ap.add_argument("--spec", default="editing", choices=list(SPECS))
    ap.add_argument("--max-pairs-per-group", type=int, default=8)
    ap.add_argument("--max-groups", type=int, default=40)
    ap.add_argument("--max-dtw-cost", type=float, default=0.55)
    ap.add_argument("--n-figures", type=int, default=6)
    ap.add_argument("--no-identity", action="store_true",
                    help="skip ArcFace (falls back to no identity gate)")
    ap.add_argument("--au", action="store_true", help="add M3 (OpenFace AU) to the gate")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    clips = load_cremad_clips(args.data)
    print(f"corpus: {len(clips)} clips, {len({c.person for c in clips})} actors")
    au_ex = None
    if args.au:
        try:
            from expverify.au import AUExtractor
            au_ex = AUExtractor()
        except Exception as e:  # noqa: BLE001
            print(f"M3 unavailable ({e}); running without AU")
    extractor = Extractor()
    corpus = build_corpus(clips, args.cache, extractor, au_extractor=au_ex)

    spec = copy.copy(SPECS[args.spec])
    weights = Weights()
    try:
        calib = Calibration.load(args.calibration)
        spec = apply_to_spec(spec, calib)
        weights = calib.weights()
        print(f"applied calibration: max_deform={spec.max_deform:.4f} "
              f"max_bs={spec.max_bs:.4f} min_pass_rate={spec.min_pass_rate:.3f}")
    except Exception as e:  # noqa: BLE001
        print(f"no calibration ({e}); using spec defaults")

    arcface = None
    if not args.no_identity:
        try:
            arcface = ArcFace()
            print("ArcFace identity gate: enabled")
        except Exception as e:  # noqa: BLE001
            print(f"ArcFace unavailable ({e}); identity gate disabled")

    by_clip = {c.stem: c for c in clips}
    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for c in clips:
        if c.meta["emotion"] == "NEU":
            continue
        if c.stem in corpus.descs:
            groups[(c.meta["sentence"], c.meta["emotion"], c.meta["intensity"])].append(c.stem)
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    keys = sorted(groups)
    rng.shuffle(keys)
    keys = keys[:args.max_groups]
    print(f"groups (sentence, emotion, intensity) with >=2 actors: {len(groups)}; "
          f"using {len(keys)}")

    frame_cache: dict[str, list[np.ndarray]] = {}
    emb_cache: dict[str, np.ndarray | None] = {}

    def frames_of(stem: str) -> list[np.ndarray]:
        if stem not in frame_cache:
            if len(frame_cache) > 24:
                frame_cache.clear()
            frame_cache[stem] = read_video(by_clip[stem].path)[0]
        return frame_cache[stem]

    def embed(stem: str) -> np.ndarray | None:
        if arcface is None:
            return None
        if stem not in emb_cache:
            f = corpus.feats[stem]
            emb_cache[stem] = arcface.embed_video(frames_of(stem),
                                                  landmarks_in_original(f), f.ok)
        return emb_cache[stem]

    # Validate the identity gate before relying on it: if different actors are
    # not separated from the same actor at this resolution, a rejection for
    # "identity cosine" says nothing about the pair.
    id_stats: dict[str, float] = {}
    if arcface:
        by_actor: dict[str, list[str]] = defaultdict(list)
        for c in clips:
            if c.stem in corpus.descs:
                by_actor[c.person].append(c.stem)
        sample = [s for stems in by_actor.values() for s in sorted(stems)[:2]][:48]
        id_stats = identity_separation({s: embed(s) for s in sample},
                                       {s: by_clip[s].person for s in sample})
        if id_stats.get("n_diff"):
            print(f"identity check on {len(sample)} clips: "
                  f"same-actor median {id_stats.get('same_median', float('nan')):.3f}, "
                  f"different-actor median {id_stats['diff_median']:.3f} "
                  f"(q95 {id_stats['diff_q95']:.3f}), "
                  f"EER threshold {id_stats.get('eer_threshold', float('nan')):.3f}")
            if spec.max_identity_cos is not None and \
                    id_stats["diff_q95"] > spec.max_identity_cos:
                thr = float(np.quantile([id_stats["diff_q95"],
                                         id_stats.get("eer_threshold", 0.0)], 1.0))
                print(f"  different actors are not below max_identity_cos="
                      f"{spec.max_identity_cos}; raising it to {thr:.3f} so the gate "
                      f"rejects same-person pairs rather than everything")
                spec.max_identity_cos = thr

    results, figures = [], 0
    manifest = ManifestWriter(out / "manifest.jsonl")
    n_align_fail = 0

    for key in tqdm(keys, desc="groups", ncols=88):
        members = groups[key]
        combos = list(itertools.combinations(sorted(members), 2))
        rng.shuffle(combos)
        for a, b in combos[:args.max_pairs_per_group]:
            da, db = corpus.descs[a], corpus.descs[b]
            fa, fb = corpus.feats[a], corpus.feats[b]
            wav_a = by_clip[a].path.with_suffix(".wav")
            wav_b = by_clip[b].path.with_suffix(".wav")
            if not (wav_a.exists() and wav_b.exists()):
                continue
            try:
                align, cost = align_videos(str(wav_a), str(wav_b), fa.fps, fb.fps,
                                           len(da), len(db))
            except Exception as e:  # noqa: BLE001
                print(f"  ! dtw {a}/{b}: {e}")
                continue
            if align.shape[0] < 8 or cost > args.max_dtw_cost:
                n_align_fail += 1
                continue

            pd = pose_delta(fa.pose[align[:, 0]], fb.pose[align[:, 1]])
            r = verify_pair(
                da, db, spec, align=align, weights=weights,
                identity_cos=cosine(embed(a), embed(b)),
                pose_delta=pd,
                extra={"group": "_".join(key), "dtw_cost": cost,
                       "emotion": key[1], "intensity": key[2], "sentence": key[0]},
            )
            results.append(r)
            manifest.write(r, group=list(key), dtw_cost=cost)
            if r.accepted and figures < args.n_figures:
                plot_pair(r, frames_of(a), frames_of(b), out / f"pair_{a}__{b}.png", spec)
                figures += 1

    manifest.close()
    if not results and figures == 0:
        print("no pairs evaluated")
        return 1

    # A rejected example is worth showing too: it is the evidence that the gate bites.
    for r in results:
        if not r.accepted and r.frames.get("t_ref") and figures < args.n_figures + 2:
            plot_pair(r, frames_of(r.ref), frames_of(r.tgt),
                      out / f"rejected_{r.ref}__{r.tgt}.png", spec)
            figures += 1
            break

    md = summarize(results, f"Demo 2 - CREMA-D cross-actor pairs (spec={spec.name})")
    extra = [f"\n- DTW alignments rejected (cost > {args.max_dtw_cost} or too short): "
             f"**{n_align_fail}**\n"]
    if id_stats.get("n_diff"):
        extra.append(
            "\n### identity gate validation (ArcFace on this footage)\n\n"
            "| | cosine |\n| --- | ---: |\n"
            f"| same actor, median | {id_stats.get('same_median', float('nan')):.3f} |\n"
            f"| different actors, median | {id_stats['diff_median']:.3f} |\n"
            f"| different actors, q95 | {id_stats['diff_q95']:.3f} |\n"
            f"| equal-error threshold | {id_stats.get('eer_threshold', float('nan')):.3f} |\n"
            f"| gate in use (max_identity_cos) | {spec.max_identity_cos} |\n")
    write_markdown(out / "summary.md", [md, *extra])
    write_schema(out / "manifest.schema.md")
    print("\n" + md)
    print(f"manifest -> {out / 'manifest.jsonl'} ({manifest.n} rows)")
    print(f"figures  -> {figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
