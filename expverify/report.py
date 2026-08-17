"""Outputs: manifest.jsonl, per-pair figures, and a markdown summary.

Every pair is recorded with the full metric vector and, when rejected, the
reason. Rejections are the more valuable half of the record: they are what makes
the acceptance rate interpretable and what turns the pipeline into something
auditable rather than a black box that emits a filtered list.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .verify import PairResult


class ManifestWriter:
    def __init__(self, path: str | Path, include_frames: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("w", encoding="utf-8")
        self.include_frames = include_frames
        self.n = 0

    def write(self, r: PairResult, **extra: Any) -> None:
        d = json.loads(r.to_json(include_frames=self.include_frames))
        d.update(extra)
        self.fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        self.n += 1

    def close(self) -> None:
        self.fh.close()

    def __enter__(self) -> "ManifestWriter":
        return self

    def __exit__(self, *a: Any) -> None:
        self.close()


def _reason_key(reason: str) -> str:
    # Longest / most specific patterns first: "accepted testable frames" also
    # contains "testable frames", and matching the short one first would merge
    # two very different failures into one misleading row.
    for key in ("accepted testable frames", "testable frames", "frame pass rate",
                "identity cosine", "pose delta", "background similarity",
                "drifted from its source", "driver identity leaked",
                "no jointly valid frames", "alignment"):
        if key in reason:
            return key
    return reason.split("(")[0].strip()[:48]


def summarize(results: list[PairResult], title: str = "expverify run") -> str:
    n = len(results)
    acc = [r for r in results if r.accepted]
    lines = [f"## {title}", "",
             f"- candidate pairs: **{n}**",
             f"- accepted: **{len(acc)}** ({100.0 * len(acc) / max(n, 1):.1f}%)"]
    if n - len(acc):
        counts = Counter(_reason_key(x) for r in results if not r.accepted for x in r.reasons)
        lines += ["", "### rejection reasons (a pair may fail several gates)", "",
                  "| reason | pairs |", "| --- | ---: |"]
        lines += [f"| {k} | {v} |" for k, v in counts.most_common()]
    if acc:
        def col(k: str) -> list[float]:
            return [r.summary[k] for r in acc if r.summary.get(k) is not None]
        lines += ["", "### accepted-pair metrics (median)", "",
                  "| metric | median |", "| --- | ---: |"]
        for k in ("pass_rate", "med_d_bs", "med_d_deform", "med_d_region",
                  "med_d_gaze", "med_d_au", "med_energy", "rank1_rate", "identity_cos"):
            v = col(k)
            if v:
                lines.append(f"| {k} | {np.median(v):.4f} |")
    return "\n".join(lines) + "\n"


def plot_pair(result: PairResult, ref_frames: list[np.ndarray] | None,
              tgt_frames: list[np.ndarray] | None, path: str | Path,
              spec: Any = None, n_show: int = 5) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = result.frames
    if not f or not f.get("t_ref"):
        return
    t = np.asarray(f["t_ref"])
    acc = np.asarray(f["accepted"], dtype=bool)
    testable = np.asarray(f["testable"], dtype=bool)

    have_img = ref_frames is not None and tgt_frames is not None
    rows = 3 + (2 if have_img else 0)
    fig = plt.figure(figsize=(13, 2.6 * rows))
    gs = fig.add_gridspec(rows, n_show, hspace=0.55, wspace=0.06)

    if have_img:
        pick = np.flatnonzero(acc)
        if pick.size < n_show:
            pick = np.arange(len(t))
        pick = pick[np.linspace(0, pick.size - 1, min(n_show, pick.size)).astype(int)]
        for c, i in enumerate(pick):
            for r, (frames, idx, lab) in enumerate((
                    (ref_frames, int(f["t_ref"][i]), "ref"),
                    (tgt_frames, int(f["t_tgt"][i]), "tgt"))):
                ax = fig.add_subplot(gs[r, c])
                if 0 <= idx < len(frames):
                    ax.imshow(frames[idx][:, :, ::-1])
                ax.set_xticks([]); ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(lab, fontsize=11)
                ax.set_title(f"f{idx}", fontsize=8)

    base = 2 if have_img else 0

    ax = fig.add_subplot(gs[base, :])
    ax.plot(t, f["d_deform"], label="d_deform (cross)", lw=1.4)
    ax.plot(t, f["d_self"], label="d_self (own +/-k)", lw=1.1, ls="--", color="gray")
    if spec is not None:
        ax.axhline(spec.max_deform, color="crimson", lw=0.9, ls=":", label="threshold")
    ax.fill_between(t, 0, 1, where=~testable, transform=ax.get_xaxis_transform(),
                    color="0.9", zorder=0, label="untestable (static)")
    ax.set_ylabel("interocular units"); ax.legend(fontsize=8, ncol=4)
    ax.set_title(f"{result.ref}  vs  {result.tgt}   "
                 f"[{'ACCEPT' if result.accepted else 'REJECT'}]", fontsize=11)

    ax = fig.add_subplot(gs[base + 1, :])
    ratio = np.array([np.nan if v is None else v for v in f["ratio"]], dtype=float)
    ax.plot(t, ratio, lw=1.3, color="tab:purple")
    ax.axhline(1.0, color="0.5", lw=0.8)
    if spec is not None:
        ax.axhline(spec.max_ratio, color="crimson", lw=0.9, ls=":")
    ax.set_ylabel("d_cross / d_self")
    top = max(1.2 * getattr(spec, "max_ratio", 1.0) if spec is not None else 1.0,
              np.nanquantile(ratio, 0.9) if np.isfinite(ratio).any() else 4.0)
    ax.set_ylim(0, float(top))
    ax.legend(["ratio", "parity", "max_ratio"], fontsize=8, ncol=3)

    ax = fig.add_subplot(gs[base + 2, :])
    ax.step(t, np.asarray(f["argmin_offset"]), where="mid", lw=1.2, color="tab:green")
    ax.axhspan(-1, 1, color="tab:green", alpha=0.12)
    ax.set_ylabel("rank-1 offset"); ax.set_xlabel("reference frame")
    ax.set_ylim(-8, 8)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def write_markdown(path: str | Path, sections: Iterable[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(sections), encoding="utf-8")


SCHEMA = """# manifest.jsonl schema

One JSON object per candidate pair, accepted or not. Rejected rows are kept:
without them the acceptance rate is unauditable, and they are themselves the
calibrated hard negatives the dataset ships with.

## top level

| field | type | meaning |
| --- | --- | --- |
| `ref`, `tgt` | str | clip names of the two sides |
| `ref_person`, `tgt_person` | str | identity labels used for neutral estimation |
| `spec` | str | dataset contract applied (`reference` / `editing`) |
| `accepted` | bool | conjunction of every gate below |
| `reasons` | list[str] | human-readable rejections; empty iff accepted |
| `summary` | object | per-pair aggregates (below) |
| `frames` | object | per-frame arrays, all the same length (below) |

## summary

| field | meaning |
| --- | --- |
| `n_frames` / `n_pass` / `n_testable` / `n_pass_testable` | frame counts after alignment |
| `pass_rate` | `n_pass / n_frames` |
| `gate_pass_rates` | per-gate frame pass rate; the smallest entry is the binding constraint |
| `med_d_bs` | M1, weighted mean abs. difference over ARKit blendshape features |
| `med_d_deform` | M2, RMS landmark-deformation disagreement, interocular units |
| `med_d_region` | worst region's disagreement divided by that region's own motion |
| `med_d_gaze` | signed gaze channels |
| `med_d_au` | M3, OpenFace AU activations (null when M3 is off) |
| `med_energy` | expressiveness of the less expressive side |
| `med_ratio` | cross-identity distance / same-video +/-k distance; < 1 means the cross match beats the temporal neighbour |
| `rank1_rate` | fraction of frames whose nearest target frame is the aligned one |
| `identity_cos` | ArcFace cosine between the two people (must be *low*) |
| `pose_delta_med`, `bg_hist` | non-expression dissimilarity evidence |
| `source_identity_cos`, `driver_identity_cos` | generation leakage guards (Track A only) |

## frames

Arrays indexed by aligned frame: `t_ref`, `t_tgt`, `d_bs`, `d_deform`, `d_gaze`,
`d_au`, `d_region`, `d_self`, `ratio`, `argmin_offset`, `testable`, `energy`,
`accepted`, and the individual gate booleans `g_bs`, `g_deform`, `g_gaze`,
`g_region`, `g_rank1`, `g_ratio`, `g_energy`, `g_au`.

`argmin_offset` is the rank-1 result: the offset of the best-matching target
frame from the aligned one. `testable` is false where the clip is too static for
the rank-1 test to mean anything; those frames are excluded rather than passed.
"""


def write_schema(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(SCHEMA, encoding="utf-8")


def calibration_markdown(thresholds: dict[str, float], report: dict[str, Any]) -> str:
    """Readable calibration record: thresholds plus the evidence behind them."""
    m = report.get("metrics") or {}
    offs = sorted({o for i in m.values() for o in (i.get("auc_by_offset") or {})}, key=int)
    out = [f"# Calibration ({report.get('positive_source', 'unknown positives')})", "",
           "Thresholds are only meaningful next to the distributions they were fitted",
           "on, so both are recorded here.", "",
           "## per-metric thresholds", "",
           "| metric | threshold | positive median | hard-negative median | AUC | recall |",
           "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for k, i in m.items():
        out.append(f"| `{k}` | {i['threshold']:.4f} | {i['pos_median']:.4f} | "
                   f"{i['hardneg_median']:.4f} | {i['auc']:.3f} | {i['recall']:.3f} |")
    if offs:
        out += ["", "## AUC vs. temporal offset of the negative", "",
                "How fine a distinction each metric can actually make. An AUC near 0.5",
                "at a small offset means the metric cannot resolve that time scale.", "",
                "| metric | " + " | ".join(f"+{o}f" for o in offs) + " |",
                "| --- |" + " ---: |" * len(offs)]
        for k, i in m.items():
            row = " | ".join(f"{(i.get('auc_by_offset') or {}).get(o, float('nan')):.3f}"
                             for o in offs)
            out.append(f"| `{k}` | {row} |")
    red = report.get("redundancy") or {}
    if red:
        out += ["", "## metric redundancy", "",
                "A conjunctive gate over two collinear metrics is one gate applied twice.",
                "", "| pair | Pearson r |", "| --- | ---: |"]
        out += [f"| {k} | {v:+.3f} |"
                for k, v in sorted(red.items(), key=lambda kv: -abs(kv[1]))]
    r1 = report.get("rank1") or {}
    if r1:
        out += ["", "## rank-1 temporal identifiability", "",
                "`shift+/-5` are negative controls: their argmin must follow the shift,",
                "otherwise the ranking test is measuring nothing.", "",
                "| case | testable frames | exact | +/-1 | median abs. offset | mean offset |",
                "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for case, d in r1.items():
            if not d.get("n_testable"):
                continue
            out.append(f"| {case} | {d['n_testable']} | {d['exact_rate_testable']:.3f} | "
                       f"{d['within_tol_rate_testable']:.3f} | "
                       f"{d['median_abs_offset_testable']:.1f} | {d['mean_offset']:+.2f} |")
    pl = report.get("pair_level") or {}
    if pl.get("min_pass_rate") is not None:
        out += ["", "## pair-level gate", "",
                f"- aligned pairs: n={pl['aligned_n']}, median pass rate "
                f"{pl['aligned_median']:.3f}",
                f"- shifted pairs: n={pl['shifted_n']}, median pass rate "
                f"{pl['shifted_median']:.3f}",
                f"- AUC {pl['auc']:.3f}, precision {pl['precision']:.3f}, "
                f"recall {pl['recall']:.3f}, "
                f"fully separated: {'yes' if pl['separated'] else 'no'}",
                f"- fitted `min_pass_rate` = {pl['min_pass_rate']:.3f}"]
    if pl.get("gate_pass_rates"):
        out += ["", "### per-gate pass rate on positives (lowest row is the binding one)",
                "", "| gate | pass rate |", "| --- | ---: |"]
        out += [f"| `{k}` | {v:.3f} |"
                for k, v in sorted(pl["gate_pass_rates"].items(), key=lambda kv: kv[1])]
    out += ["", "## all fitted thresholds", "", "| key | value |", "| --- | ---: |"]
    out += [f"| `{k}` | {v:.4f} |" for k, v in sorted(thresholds.items())]
    return "\n".join(out) + "\n"
