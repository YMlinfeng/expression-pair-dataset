"""Fetch a small CREMA-D subset without cloning the 7.5 GB repository.

CREMA-D is the fastest route to *real* cross-identity pairs: 91 actors each
speak the same 12 sentences in 6 emotions at 4 intensity levels, so a
(sentence, emotion, intensity) group contains up to 91 different people
performing the same prescribed thing. Licence is ODbL -- open, no EULA, no
registration, commercial use permitted -- which is unusual in this corner of the
field.

Individual clips are ~260 KB and are served directly from the git-LFS media
endpoint, so a useful subset is tens of megabytes rather than gigabytes.
"""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RAW = "https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master"
LFS = "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master"

SENTENCES = ["IEO", "TIE", "IOM", "IWW", "TAI", "MTI",
             "IWL", "ITH", "DFA", "ITS", "TSI", "WSI"]
EMOTIONS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
INTENSITIES = ["LO", "MD", "HI", "XX"]


def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def load_index() -> list[str]:
    rows = list(csv.DictReader(io.StringIO(fetch_text(f"{RAW}/SentenceFilenames.csv"))))
    return [r["Filename"] for r in rows]


def load_demographics() -> dict[str, dict[str, str]]:
    rows = list(csv.DictReader(io.StringIO(fetch_text(f"{RAW}/VideoDemographics.csv"))))
    return {r["ActorID"]: r for r in rows}


def parse(name: str) -> tuple[str, str, str, str]:
    actor, sentence, emotion, intensity = name.split("_")
    return actor, sentence, emotion, intensity


def download_clip(name: str, out_dir: Path, keep_flv: bool = False) -> Path | None:
    mp4 = out_dir / f"{name}.mp4"
    wav = out_dir / f"{name}.wav"
    if mp4.exists() and wav.exists() and mp4.stat().st_size > 1000:
        return mp4
    flv = out_dir / f"{name}.flv"
    try:
        urllib.request.urlretrieve(f"{LFS}/VideoFlash/{name}.flv", flv)
    except Exception as e:  # noqa: BLE001
        print(f"  ! download failed {name}: {e}", file=sys.stderr)
        return None
    ff = ffmpeg_exe()
    try:
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(flv),
                        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                        "-an", str(mp4)], check=True)
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(flv),
                        "-vn", "-ac", "1", "-ar", "16000", str(wav)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"  ! transcode failed {name}: {e}", file=sys.stderr)
        return None
    finally:
        if not keep_flv and flv.exists():
            flv.unlink()
    return mp4


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/cremad")
    ap.add_argument("--actors", type=int, default=12,
                    help="number of distinct actors (balanced by sex)")
    ap.add_argument("--sentences", nargs="+", default=["IEO", "TIE", "IOM"])
    ap.add_argument("--emotions", nargs="+", default=["ANG", "HAP", "SAD", "DIS", "FEA"])
    ap.add_argument("--intensities", nargs="+", default=["HI"])
    ap.add_argument("--with-neutral", action="store_true", default=True,
                    help="also fetch NEU_XX clips, used for per-person neutral estimation")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = load_index()
    demo = load_demographics()

    all_actors = sorted({parse(n)[0] for n in names})
    males = [a for a in all_actors if demo.get(a, {}).get("Sex") == "Male"]
    females = [a for a in all_actors if demo.get(a, {}).get("Sex") == "Female"]
    half = args.actors // 2
    actors = set(males[:half] + females[:args.actors - half])

    wanted = []
    for n in names:
        a, s, e, i = parse(n)
        if a not in actors or s not in args.sentences:
            continue
        if e == "NEU":
            if args.with_neutral:
                wanted.append(n)
        elif e in args.emotions and i in args.intensities:
            wanted.append(n)

    print(f"actors={len(actors)} clips={len(wanted)} -> {out_dir}")
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(lambda n: download_clip(n, out_dir), wanted):
            done += r is not None
    print(f"downloaded/verified {done}/{len(wanted)} clips")

    meta = out_dir / "clips.csv"
    with meta.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "actor", "sentence", "emotion", "intensity", "sex", "age", "race"])
        for n in sorted(wanted):
            if not (out_dir / f"{n}.mp4").exists():
                continue
            a, s, e, i = parse(n)
            d = demo.get(a, {})
            w.writerow([f"{n}.mp4", a, s, e, i, d.get("Sex", ""), d.get("Age", ""), d.get("Race", "")])
    print(f"wrote {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
