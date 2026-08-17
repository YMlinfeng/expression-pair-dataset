"""Threshold calibration against a hard-negative distribution.

A threshold is only meaningful relative to what it must reject. Three
distributions are collected:

  positives     clip frame t vs. frame t of its appearance-augmented twin.
                Expression is identical by construction, so this is the
                verifier's noise floor -- the best score any real pair can reach.
  hard negatives  clip frame t vs. frame t+k of that *same augmented twin*.
                Same person, same background, same lighting, same appearance
                change, same two-independent-runs noise -- differing only by k
                frames of muscle movement. These are the "差不多" cases.
  null          random frames from different people. The easy baseline; used to
                derive per-feature scales so the M1 channels can be combined.

The paired construction matters more than it looks. Measuring negatives as
d(t, t+k) *within a single extraction run* looks equivalent but is not:
FaceLandmarker's VIDEO mode carries tracking state across frames, so landmarks
within one run are temporally smoothed and their noise is correlated, while a
positive compares two independent runs and carries full independent noise.
Comparing those two distributions understates every negative and makes a
perfectly good metric look worse than chance. Both sides of the paired design
sit in the same noise regime, so the only remaining difference is the k frames
of expression change that we actually want to measure.

Thresholds are then fitted for **precision**, not accuracy: a strict verifier
should reject an ambiguous pair rather than adjudicate it. If the positive and
hard-negative distributions overlap for a metric, that metric cannot resolve the
required granularity, and the report says so instead of hiding it behind a
tuned constant.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .neutral import SeqDescriptor
from .verify import Weights, bs_distance, deform_distance, gaze_distance, region_worst
from .landmarks import REGIONS

METRICS = ("d_bs", "d_deform", "d_gaze", "d_region")


@dataclass
class Calibration:
    bs_sigma: np.ndarray
    pos: dict[str, np.ndarray] = field(default_factory=dict)
    neg: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)   # offset -> metric -> vals
    null: dict[str, np.ndarray] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)

    def weights(self) -> Weights:
        return Weights(bs=self.bs_sigma)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            Path(path).with_suffix(".npz"), bs_sigma=self.bs_sigma,
            **{f"pos__{k}": v for k, v in self.pos.items()},
            **{f"null__{k}": v for k, v in self.null.items()},
            **{f"neg__{o}__{m}": v for o, d in self.neg.items() for m, v in d.items()},
        )
        Path(path).with_suffix(".json").write_text(
            json.dumps({"thresholds": self.thresholds, "report": self.report},
                       indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "Calibration":
        npz = np.load(Path(path).with_suffix(".npz"))
        meta = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
        pos, null, neg = {}, {}, {}
        for k in npz.files:
            if k.startswith("pos__"):
                pos[k[5:]] = npz[k]
            elif k.startswith("null__"):
                null[k[6:]] = npz[k]
            elif k.startswith("neg__"):
                _, o, m = k.split("__")
                neg.setdefault(o, {})[m] = npz[k]
        return Calibration(bs_sigma=npz["bs_sigma"], pos=pos, neg=neg, null=null,
                           thresholds=meta["thresholds"], report=meta["report"])


def _frame_metrics(a: SeqDescriptor, ia: int, b: SeqDescriptor, ib: int,
                   w: np.ndarray) -> dict[str, float]:
    _, d_rg = region_worst(a.deform[ia], b.deform[ib], REGIONS)
    out = {
        "d_bs": bs_distance(a.bs[ia], b.bs[ib], w),
        "d_deform": deform_distance(a.deform[ia], b.deform[ib]),
        "d_gaze": gaze_distance(a.gaze[ia], b.gaze[ib]),
        "d_region": d_rg,
    }
    if a.au is not None and b.au is not None:
        from .au import au_distance
        out["d_au"] = au_distance(a.au[ia], b.au[ib])
    return out


def feature_scales(descs: list[SeqDescriptor], n_samples: int = 20000,
                   seed: int = 0) -> np.ndarray:
    """Per-M1-feature spread over random cross-identity frame pairs.

    Used as inverse weights so that channels with naturally large dynamic range
    do not dominate the L1, and so a threshold is portable across corpora.
    """
    rng = np.random.default_rng(seed)
    pool = [(i, t) for i, d in enumerate(descs) for t in np.flatnonzero(d.ok)]
    if len(pool) < 2:
        return np.ones(descs[0].bs.shape[1], dtype=np.float32)
    diffs = []
    for _ in range(n_samples):
        (i, t), (j, s) = pool[rng.integers(len(pool))], pool[rng.integers(len(pool))]
        if descs[i].person == descs[j].person:
            continue
        diffs.append(np.abs(descs[i].bs[t] - descs[j].bs[s]))
    if not diffs:
        return np.ones(descs[0].bs.shape[1], dtype=np.float32)
    return np.maximum(np.mean(diffs, axis=0), 1e-4).astype(np.float32)


def collect_positives(pairs: list[tuple[SeqDescriptor, SeqDescriptor]],
                      w: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, list[float]] = defaultdict(list)
    for a, b in pairs:
        n = min(len(a), len(b))
        for t in range(n):
            if not (a.ok[t] and b.ok[t]):
                continue
            for m, v in _frame_metrics(a, t, b, t, w).items():
                out[m].append(v)
    return {m: np.asarray(v, dtype=np.float32) for m, v in out.items()}


def collect_paired_negatives(pairs: list[tuple[SeqDescriptor, SeqDescriptor]],
                             offsets: tuple[int, ...],
                             w: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Hard negatives in the same noise regime as the positives.

    For each positive pair (a, b) with identical geometry, take d(a_t, b_{t+k}).
    Everything is held constant except k frames of expression change.
    """
    neg: dict[str, dict[str, list[float]]] = {str(k): defaultdict(list) for k in offsets}
    for a, b in pairs:
        n = min(len(a), len(b))
        for k in offsets:
            for t in range(n):
                s = t + k
                if s >= n or not (a.ok[t] and b.ok[s]):
                    continue
                for m, v in _frame_metrics(a, t, b, s, w).items():
                    neg[str(k)][m].append(v)
    return {o: {m: np.asarray(v, dtype=np.float32) for m, v in md.items()}
            for o, md in neg.items()}


def collect_temporal_negatives(descs: list[SeqDescriptor], offsets: tuple[int, ...],
                               w: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Within-run d(t, t+k). Kept as a diagnostic only: VIDEO-mode tracking
    smooths landmarks along a clip, so these are optimistically small and are
    not a fair reference for a cross-video threshold."""
    neg: dict[str, dict[str, list[float]]] = {str(k): defaultdict(list) for k in offsets}
    for d in descs:
        idx = np.flatnonzero(d.ok)
        for k in offsets:
            for t in idx:
                s = t + k
                if s >= len(d) or not d.ok[s]:
                    continue
                for m, v in _frame_metrics(d, int(t), d, int(s), w).items():
                    neg[str(k)][m].append(v)
    return {o: {m: np.asarray(v, dtype=np.float32) for m, v in md.items()}
            for o, md in neg.items()}


def collect_null(descs: list[SeqDescriptor], w: np.ndarray, n_samples: int = 4000,
                 seed: int = 1) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    pool = [(i, int(t)) for i, d in enumerate(descs) for t in np.flatnonzero(d.ok)]
    out: dict[str, list[float]] = defaultdict(list)
    tries = 0
    while len(out["d_bs"]) < n_samples and tries < n_samples * 10:
        tries += 1
        (i, t), (j, s) = pool[rng.integers(len(pool))], pool[rng.integers(len(pool))]
        if descs[i].person == descs[j].person:
            continue
        for m, v in _frame_metrics(descs[i], t, descs[j], s, w).items():
            out[m].append(v)
    return {m: np.asarray(v, dtype=np.float32) for m, v in out.items()}


def rank1_diagnostics(pairs: list[tuple[SeqDescriptor, SeqDescriptor]],
                      window: int = 12, k: int = 3, tol: int = 1,
                      min_self_contrast: float = 0.006,
                      shift: int = 0) -> dict[str, Any]:
    """How often the aligned frame is the nearest frame, and by how much.

    This is the measurement that decides whether the whole approach works.
    Absolute distances are contaminated by a noise floor that no threshold can
    remove, but the *ranking* of candidate frames is far more robust: a bias
    shared by every candidate cancels. `shift` runs the same test on a
    deliberately misaligned pair as a negative control -- the argmin histogram
    should move with it, otherwise the test is measuring nothing.
    """
    offsets, ratios, testable_flags = [], [], []
    for a, b in pairs:
        n = min(len(a), len(b))
        for t in range(n):
            tt = t + shift
            if tt < 0 or tt >= n or not a.ok[t]:
                continue
            lo, hi = max(0, tt - window), min(n, tt + window + 1)
            cand = [s for s in range(lo, hi) if b.ok[s]]
            if len(cand) < 3 or not b.ok[tt]:
                continue
            d = np.array([deform_distance(a.deform[t], b.deform[s]) for s in cand])
            best = cand[int(np.argmin(d))]
            offsets.append(best - tt)
            far = [j for j, s in enumerate(cand) if abs(s - tt) >= k]
            if far:
                d_far = float(d[far].min())
                d_self = float(deform_distance(a.deform[t], b.deform[tt]))
                ratios.append(d_self / max(d_far, 1e-9))
            sc = [s for s in (t - k, t + k) if 0 <= s < len(a) and a.ok[s]]
            testable_flags.append(bool(sc) and
                                  min(deform_distance(a.deform[t], a.deform[s]) for s in sc)
                                  >= min_self_contrast)
    off = np.asarray(offsets)
    rat = np.asarray(ratios)
    tst = np.asarray(testable_flags, dtype=bool)
    if off.size == 0:
        return {"n": 0}
    out = {
        "n": int(off.size),
        "exact_rate": float((off == 0).mean()),
        "within_tol_rate": float((np.abs(off) <= tol).mean()),
        "within_2_rate": float((np.abs(off) <= 2).mean()),
        "median_abs_offset": float(np.median(np.abs(off))),
        "mean_offset": float(off.mean()),
        "median_ratio": float(np.median(rat)) if rat.size else None,
        "testable_rate": float(tst.mean()) if tst.size else None,
    }
    # Frames inside a static segment are genuinely equal to their neighbours, so
    # a ranking test cannot succeed there by construction. Scoring them as
    # failures understates the metric; the testable subset is the honest number.
    if tst.size == off.size and tst.any():
        o = off[tst]
        out.update({
            "n_testable": int(tst.sum()),
            "exact_rate_testable": float((o == 0).mean()),
            "within_tol_rate_testable": float((np.abs(o) <= tol).mean()),
            "within_2_rate_testable": float((np.abs(o) <= 2).mean()),
            "median_abs_offset_testable": float(np.median(np.abs(o))),
        })
    return out


def pair_rank1_rates(pairs: list[tuple[SeqDescriptor, SeqDescriptor]],
                     window: int = 12, k: int = 3, tol: int = 1,
                     min_self_contrast: float = 0.006, shift: int = 0) -> np.ndarray:
    """Per-pair fraction of testable frames whose nearest target frame is aligned.

    Per-frame the rank-1 test is noisy; per pair it is not. Fifty weakly
    informative frames aggregate into a decisive verdict, which is why the
    accept/reject decision is made at the pair level and never frame by frame.
    """
    rates = []
    for a, b in pairs:
        n = min(len(a), len(b))
        hits, tot = 0, 0
        for t in range(n):
            tt = t + shift
            if tt < 0 or tt >= n or not (a.ok[t] and b.ok[tt]):
                continue
            sc = [s for s in (t - k, t + k) if 0 <= s < len(a) and a.ok[s]]
            if not sc or min(deform_distance(a.deform[t], a.deform[s])
                             for s in sc) < min_self_contrast:
                continue
            lo, hi = max(0, tt - window), min(n, tt + window + 1)
            cand = [s for s in range(lo, hi) if b.ok[s]]
            if len(cand) < 3:
                continue
            d = [deform_distance(a.deform[t], b.deform[s]) for s in cand]
            tot += 1
            hits += abs(cand[int(np.argmin(d))] - tt) <= tol
        if tot >= 5:
            rates.append(hits / tot)
    return np.asarray(rates, dtype=np.float32)


def fit_threshold(pos: np.ndarray, neg: np.ndarray, precision: float = 0.95,
                  min_recall: float = 0.5) -> tuple[float, dict[str, float]]:
    """Largest threshold whose precision on {pos vs neg} still meets the target.

    `min_recall` is not a nicety. When the two distributions overlap heavily,
    the highest-precision operating point sits far below the positive median and
    rejects almost every true pair -- a threshold with 1.6% recall is not strict,
    it is broken. If the target precision is unreachable at usable recall the
    function falls back to the positive q90 and says so, so the failure is
    visible in the report instead of silently emptying the dataset.
    """
    if pos.size == 0 or neg.size == 0:
        return float("inf"), {"auc": float("nan"), "recall": 0.0, "precision": 0.0}
    cands = np.unique(np.concatenate([pos, neg]))
    if cands.size > 4000:
        cands = np.quantile(cands, np.linspace(0, 1, 4000))
    best_thr, best = None, {"auc": 0.0, "recall": 0.0, "precision": 0.0}
    for thr in cands:
        tp = float((pos <= thr).sum())
        fp = float((neg <= thr).sum())
        if tp + fp == 0:
            continue
        p = tp / (tp + fp)
        r = tp / pos.size
        if p >= precision and r >= min_recall and r > best["recall"]:
            best_thr, best = float(thr), {"precision": p, "recall": r}
    # separation diagnostics, independent of the chosen operating point
    labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    scores = -np.concatenate([pos, neg])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, labels.size + 1)
    n1, n0 = float(labels.sum()), float((1 - labels).sum())
    auc = (ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float("nan")
    best["auc"] = float(auc)
    best["pos_q95"] = float(np.quantile(pos, 0.95))
    best["neg_q05"] = float(np.quantile(neg, 0.05))
    best["separated"] = bool(best["pos_q95"] < best["neg_q05"])
    if best_thr is None:
        # No operating point reaches the target precision at usable recall:
        # fall back to the noise floor and flag it loudly in the report.
        best_thr = float(np.quantile(pos, 0.90))
        best["precision"] = float((pos <= best_thr).sum() /
                                  max(1.0, (pos <= best_thr).sum() + (neg <= best_thr).sum()))
        best["recall"] = float((pos <= best_thr).mean())
        best["target_precision_unreachable"] = True
    return best_thr, best


def collect_ratios(pairs: list[tuple[SeqDescriptor, SeqDescriptor]],
                   window: int = 12, k: int = 3, min_self_contrast: float = 0.006,
                   shift: int = 0) -> np.ndarray:
    """d(ref_t, tgt_aligned) / d(ref_t, ref_{t+/-k}) over testable frames."""
    out = []
    for a, b in pairs:
        n = min(len(a), len(b))
        for t in range(n):
            tt = t + shift
            if tt < 0 or tt >= n or not (a.ok[t] and b.ok[tt]):
                continue
            sc = [s for s in (t - k, t + k) if 0 <= s < len(a) and a.ok[s]]
            if not sc:
                continue
            d_self = min(deform_distance(a.deform[t], a.deform[s]) for s in sc)
            if d_self < min_self_contrast:
                continue
            out.append(deform_distance(a.deform[t], b.deform[tt]) / max(d_self, 1e-9))
    return np.asarray(out, dtype=np.float32)


def calibrate(pos_pairs: list[tuple[SeqDescriptor, SeqDescriptor]],
              descs: list[SeqDescriptor],
              offsets: tuple[int, ...] = (3, 5, 10),
              precision: float = 0.95,
              hard_offset: int = 3,
              energy_percentile: float = 0.35) -> Calibration:
    sigma = feature_scales(descs)
    w = Weights(bs=sigma).bs_weights(descs[0].bs.shape[1])
    pos = collect_positives(pos_pairs, w)
    neg = collect_paired_negatives(pos_pairs, offsets, w)
    within = collect_temporal_negatives(descs, offsets, w)
    null = collect_null(descs, w)

    thresholds: dict[str, float] = {}
    report: dict[str, Any] = {"n_pos": int(pos["d_bs"].size),
                              "precision_target": precision,
                              "hard_negative_offset": hard_offset,
                              "hard_negative_kind": "paired (same augmented twin, shifted)",
                              "within_run_median": {
                                  m: float(np.median(within[str(hard_offset)][m]))
                                  for m in METRICS if within[str(hard_offset)][m].size},
                              "metrics": {}}
    hard = neg[str(hard_offset)]
    active = [m for m in (*METRICS, "d_au") if pos.get(m) is not None and pos[m].size]
    for m in active:
        thr, info = fit_threshold(pos[m], hard[m], precision)
        thresholds[m] = thr
        info["null_q05"] = float(np.quantile(null[m], 0.05)) if null[m].size else None
        info["pos_median"] = float(np.median(pos[m]))
        info["hardneg_median"] = float(np.median(hard[m]))
        info["null_median"] = float(np.median(null[m])) if null[m].size else None
        info["threshold"] = thr
        info["n_hardneg"] = int(hard[m].size)
        # How AUC decays as the negative gets temporally closer is the honest
        # statement of what granularity this metric can actually resolve.
        info["auc_by_offset"] = {o: fit_threshold(pos[m], neg[o][m], precision)[1]["auc"]
                                 for o in neg if neg[o].get(m) is not None and neg[o][m].size}
        report["metrics"][m] = info

    # Independence check: a conjunctive gate over two highly correlated metrics
    # is one gate wearing two hats.
    pool = {m: np.concatenate([pos[m], hard[m]]) for m in active
            if pos[m].size and hard[m].size}
    report["redundancy"] = {
        f"{a}~{b}": float(np.corrcoef(pool[a], pool[b])[0, 1])
        for i, a in enumerate(sorted(pool)) for b in sorted(pool)[i + 1:]
        if pool[a].size == pool[b].size
    }

    report["rank1"] = {"aligned": rank1_diagnostics(pos_pairs)}
    for s in (-5, 5):
        report["rank1"][f"shift{s:+d}"] = rank1_diagnostics(pos_pairs, shift=s)

    # max_ratio must be fitted, not assumed. A hand-picked 0.8 demands that the
    # cross-identity match beat the video's own temporal contrast outright,
    # which even construction-guaranteed positives do not achieve (their median
    # ratio sits near 1.0); such a constant silently rejects everything.
    r_pos = collect_ratios(pos_pairs)
    r_neg = np.concatenate([collect_ratios(pos_pairs, shift=s) for s in (-10, -5, 5, 10)])
    if r_pos.size and r_neg.size:
        thr_ratio, info_ratio = fit_threshold(r_pos, r_neg, precision)
        thresholds["max_ratio"] = thr_ratio
        report["ratio"] = {"threshold": thr_ratio, "pos_median": float(np.median(r_pos)),
                           "neg_median": float(np.median(r_neg)), **info_ratio}

    # Expressiveness gate as a corpus percentile: near-neutral frames match
    # trivially, so accepting them inflates every number while teaching nothing.
    # The gate is applied to min(ref, tgt), so a percentile p costs noticeably
    # more than (1 - p) of frames; this is a yield/quality knob, not a
    # discrimination one, and it is reported separately for that reason.
    energies = np.concatenate([d.energy[d.ok] for d in descs if d.ok.any()])
    thresholds["min_energy"] = float(np.quantile(energies, energy_percentile))
    report["energy"] = {"percentile": energy_percentile,
                        "min_energy": thresholds["min_energy"],
                        "corpus_median": float(np.median(energies))}

    report["rank1_only_pair_level"] = _rank1_pair_summary(pos_pairs)
    return Calibration(bs_sigma=sigma, pos=pos, neg=neg, null=null,
                       thresholds=thresholds, report=report)


def _rank1_pair_summary(pos_pairs) -> dict[str, Any]:
    aligned = pair_rank1_rates(pos_pairs)
    shifted = np.concatenate([pair_rank1_rates(pos_pairs, shift=s) for s in (-10, -5, 5, 10)])
    if not (aligned.size and shifted.size):
        return {}
    thr_neg, info = fit_threshold(-aligned, -shifted, 0.95, min_recall=0.5)
    return {"aligned_n": int(aligned.size), "aligned_median": float(np.median(aligned)),
            "aligned_q05": float(np.quantile(aligned, 0.05)),
            "shifted_n": int(shifted.size), "shifted_median": float(np.median(shifted)),
            "shifted_q95": float(np.quantile(shifted, 0.95)),
            "auc": info["auc"], "rate_threshold": float(-thr_neg)}


def calibrate_pair_level(spec, pos_pairs: list[tuple[SeqDescriptor, SeqDescriptor]],
                         weights: Weights, shifts: tuple[int, ...] = (-10, -5, 5, 10),
                         precision: float = 0.95) -> dict[str, Any]:
    """Fit `min_pass_rate` by running the *actual* verifier, not a proxy.

    The frame gate is conjunctive over seven conditions; a pass rate fitted from
    the rank-1 criterion alone would not describe it. Driving the real
    `verify_pair` on aligned positives and on deliberately shifted copies makes
    the fitted threshold consistent with the code that will enforce it.
    """
    from .verify import verify_pair

    def rates(shift: int) -> np.ndarray:
        out = []
        for a, b in pos_pairs:
            n = min(len(a), len(b))
            ref_idx = np.arange(max(0, -shift), min(n, n - shift))
            if ref_idx.size < 10:
                continue
            align = np.stack([ref_idx, ref_idx + shift], axis=1)
            r = verify_pair(a, b, spec, align=align, weights=weights)
            if r.summary.get("n_frames", 0) >= 10:
                out.append(r.summary["pass_rate"])
        return np.asarray(out, dtype=np.float32)

    # Which single gate is binding on true positives? Without this, a low pass
    # rate is unattributable and the natural reaction is to loosen everything.
    attribution: dict[str, float] = {}
    for a, b in pos_pairs:
        r = verify_pair(a, b, spec, weights=weights)
        for k, v in r.summary.get("gate_pass_rates", {}).items():
            attribution[k] = attribution.get(k, 0.0) + v
    attribution = {k: v / max(len(pos_pairs), 1) for k, v in attribution.items()}

    aligned = rates(0)
    shifted = np.concatenate([rates(s) for s in shifts]) if shifts else np.zeros(0)
    if not aligned.size:
        return {"gate_pass_rates": attribution}
    if not shifted.size:
        return {"aligned_median": float(np.median(aligned)), "gate_pass_rates": attribution}
    thr_neg, info = fit_threshold(-aligned, -shifted, precision, min_recall=0.5)
    return {
        "min_pass_rate": float(-thr_neg), "gate_pass_rates": attribution,
        "aligned_n": int(aligned.size), "aligned_median": float(np.median(aligned)),
        "aligned_q05": float(np.quantile(aligned, 0.05)),
        "shifted_n": int(shifted.size), "shifted_median": float(np.median(shifted)),
        "shifted_q95": float(np.quantile(shifted, 0.95)),
        "auc": info["auc"], "recall": info["recall"], "precision": info["precision"],
        "separated": bool(np.quantile(shifted, 0.95) < np.quantile(aligned, 0.05)),
    }


def apply_to_spec(spec, calib: Calibration):
    """Return a copy of `spec` with calibrated expression thresholds."""
    import copy
    s = copy.copy(spec)
    s.max_bs = calib.thresholds.get("d_bs", s.max_bs)
    s.max_deform = calib.thresholds.get("d_deform", s.max_deform)
    s.max_gaze = calib.thresholds.get("d_gaze", s.max_gaze)
    s.max_region = calib.thresholds.get("d_region", s.max_region)
    if "d_au" in calib.thresholds:
        s.max_au = calib.thresholds["d_au"]
    for key, attr in (("max_ratio", "max_ratio"), ("min_energy", "min_energy"),
                      ("min_pass_rate", "min_pass_rate")):
        if key in calib.thresholds:
            setattr(s, attr, calib.thresholds[key])
    s.name = f"{spec.name}+calibrated"
    return s
