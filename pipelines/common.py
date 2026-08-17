"""Shared corpus construction: video -> features -> per-person neutral -> descriptors."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from expverify.descriptors import (ChannelPlan, build_channel_plan, fit_rigid_reference,
                                   rigid_samples, set_reference)
from expverify.landmarks import (Extractor, VideoFeats, crop_frames, extract_cached,
                                 read_video)
from expverify.au import neutral_au
from expverify.neutral import Neutral, SeqDescriptor, describe, estimate_neutral

REFERENCE_PATH = Path(__file__).resolve().parent.parent / "models" / "rigid_reference.npy"


def ensure_reference(feats: list[VideoFeats], path: Path = REFERENCE_PATH,
                     refit: bool = False) -> np.ndarray:
    """Fit (or load) the shared rigid reference and install it globally.

    One reference is used for every clip in every track, which is what puts two
    different people's deformation fields into a common frame.
    """
    if path.exists() and not refit:
        ref = np.load(path)
    else:
        samples = np.concatenate([rigid_samples(f.landmarks, f.ok) for f in feats
                                  if f.n_valid > 0], axis=0)
        if samples.shape[0] < 8:
            raise RuntimeError("not enough valid frames to fit a rigid reference")
        ref = fit_rigid_reference(samples)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, ref)
    set_reference(ref)
    return ref


def au_for_feats(au_ex, frames: list[np.ndarray], f: VideoFeats) -> np.ndarray:
    """(T, 8) AU activations on the same face crop the landmarks came from.

    The crop is reconstructed from `crop_box` rather than re-detected, so M3 and
    M1/M2 are guaranteed to be reading the same pixels.
    """
    if f.crop_box is not None:
        frames = crop_frames(frames, f.crop_box, max(f.width, 1))
    return au_ex.run(frames[:len(f)], f.ok[:len(frames)])


def au_cached(au_ex, video: str | Path, f: VideoFeats, cache_dir: str | Path,
              max_frames: int | None = None) -> np.ndarray:
    cp = Path(cache_dir) / (Path(video).stem + "__au.npy")
    if cp.exists():
        a = np.load(cp)
        if a.shape[0] == len(f):
            return a
    frames, _ = read_video(video, max_frames=max_frames)
    a = au_for_feats(au_ex, frames, f)
    cp.parent.mkdir(parents=True, exist_ok=True)
    np.save(cp, a)
    return a


@dataclass
class Clip:
    path: Path
    person: str
    meta: dict[str, str]

    @property
    def stem(self) -> str:
        return self.path.stem


@dataclass
class Corpus:
    clips: list[Clip]
    feats: dict[str, VideoFeats]
    plan: ChannelPlan
    neutrals: dict[str, Neutral]
    descs: dict[str, SeqDescriptor]

    def desc_list(self) -> list[SeqDescriptor]:
        return list(self.descs.values())


def load_cremad_clips(data_dir: str | Path) -> list[Clip]:
    data_dir = Path(data_dir)
    rows = list(csv.DictReader((data_dir / "clips.csv").open()))
    return [Clip(path=data_dir / r["file"], person=r["actor"], meta=r)
            for r in rows if (data_dir / r["file"]).exists()]


def build_corpus(clips: list[Clip], cache_dir: str | Path,
                 extractor: Extractor | None = None,
                 max_frames: int | None = None,
                 face_crop: bool = True, crop_size: int = 512,
                 au_extractor=None, desc: str = "extract") -> Corpus:
    extractor = extractor or Extractor()
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    feats: dict[str, VideoFeats] = {}
    for c in tqdm(clips, desc=desc, ncols=88):
        try:
            feats[c.stem] = extract_cached(extractor, c.path, cache_dir,
                                           max_frames=max_frames, face_crop=face_crop,
                                           crop_size=crop_size)
        except Exception as e:  # noqa: BLE001
            print(f"  ! {c.stem}: {e}")
    clips = [c for c in clips if c.stem in feats]
    if not clips:
        raise RuntimeError("no clips produced features")

    aus: dict[str, np.ndarray] = {}
    if au_extractor is not None:
        for c in tqdm(clips, desc="au", ncols=88):
            try:
                aus[c.stem] = au_cached(au_extractor, c.path, feats[c.stem], cache_dir,
                                        max_frames=max_frames)
            except Exception as e:  # noqa: BLE001
                print(f"  ! au {c.stem}: {e}")

    ensure_reference(list(feats.values()))
    plan = build_channel_plan(next(iter(feats.values())).bs_names)

    by_person: dict[str, list[VideoFeats]] = {}
    # Kept as (feats, au) tuples so a clip whose AU extraction failed cannot
    # shift the pairing and attach one clip's AU rows to another clip's mask.
    au_by_person: dict[str, list[tuple[VideoFeats, np.ndarray]]] = {}
    for c in clips:
        by_person.setdefault(c.person, []).append(feats[c.stem])
        if c.stem in aus:
            au_by_person.setdefault(c.person, []).append((feats[c.stem], aus[c.stem]))

    neutrals: dict[str, Neutral] = {}
    for person, fl in by_person.items():
        try:
            n = estimate_neutral(fl, plan, person)
        except ValueError as e:
            print(f"  ! neutral for {person}: {e}")
            continue
        if person in au_by_person:
            rows = [(f.ok[:len(a)], a[:len(f.ok)]) for f, a in au_by_person[person]]
            n.au = neutral_au(np.concatenate([a for _, a in rows]),
                              np.concatenate([m for m, _ in rows]))
        neutrals[person] = n

    descs: dict[str, SeqDescriptor] = {}
    for c in clips:
        if c.person not in neutrals:
            continue
        descs[c.stem] = describe(feats[c.stem], plan, neutrals[c.person],
                                 au=aus.get(c.stem))

    return Corpus(clips=clips, feats=feats, plan=plan, neutrals=neutrals, descs=descs)


def valid_fraction(corpus: Corpus) -> float:
    tot = sum(len(d) for d in corpus.descs.values())
    good = sum(int(d.ok.sum()) for d in corpus.descs.values())
    return good / max(tot, 1)
