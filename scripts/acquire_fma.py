"""Local equivalent of notebooks/01_fma_acquire_and_curate.ipynb -- no GPU needed for
this step, so it runs here instead of Colab. Downloads fma_small + fma_metadata,
stratified-samples ~1,400 tracks across the 8 balanced genres, extracts just those
MP3s into data/audio/, and writes data/fma_metadata/curated_tracks.csv.

Sampling logic matches the notebook exactly, including the fix for a real pandas
groupby/apply gotcha (grouping by a column reference silently drops that column from
the result) caught during a local dry-run before this was pointed at the real 7.2GB zip.
"""

import os
import sys
import zipfile

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sonic_explorer.config import AUDIO_DIR, DATA_DIR, METADATA_DIR

STAGING_DIR = DATA_DIR / "_downloads"
FMA_SMALL_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
FMA_METADATA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
TRACKS_PER_GENRE = 175  # ~175 x 8 genres ~= 1,400 tracks, within the spec's 1,000-2,000 target


def download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"{dest} already present, skipping download", flush=True)
        return
    print(f"Downloading {url} -> {dest}", flush=True)
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    next_report_pct = 5
    with open(dest + ".part", "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            downloaded += len(chunk)
            if total and downloaded / total * 100 >= next_report_pct:
                print(f"  {os.path.basename(dest)}: {downloaded / total * 100:.0f}% "
                      f"({downloaded / 1e6:.0f}MB / {total / 1e6:.0f}MB)", flush=True)
                next_report_pct += 5
    os.rename(dest + ".part", dest)
    print(f"Done: {dest}", flush=True)


def fma_zip_path(track_id: int) -> str:
    return f"fma_small/{track_id // 1000:03d}/{track_id:06d}.mp3"


def main():
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    metadata_zip = str(STAGING_DIR / "fma_metadata.zip")
    small_zip = str(STAGING_DIR / "fma_small.zip")
    download(FMA_METADATA_URL, metadata_zip)
    download(FMA_SMALL_URL, small_zip)

    print("Parsing tracks.csv ...", flush=True)
    with zipfile.ZipFile(metadata_zip) as zf:
        zf.extract("fma_metadata/tracks.csv", str(STAGING_DIR))
    tracks = pd.read_csv(STAGING_DIR / "fma_metadata" / "tracks.csv", index_col=0, header=[0, 1])

    small = tracks[tracks[("set", "subset")] == "small"].copy()
    small = small[small[("track", "genre_top")].notna()]
    print(f"{len(small)} fma_small tracks with a genre_top label", flush=True)

    genre_series = small[("track", "genre_top")]
    curated_parts = [
        group.sample(n=min(TRACKS_PER_GENRE, len(group)), random_state=42)
        for _, group in small.groupby(genre_series)
    ]
    curated = pd.concat(curated_parts)
    print(f"Curated {len(curated)} tracks:", flush=True)
    print(curated[("track", "genre_top")].value_counts(), flush=True)

    print("Extracting curated MP3s ...", flush=True)
    manifest_rows = []
    with zipfile.ZipFile(small_zip) as zf:
        for i, (track_id, row) in enumerate(curated.iterrows()):
            zip_path = fma_zip_path(track_id)
            dest_name = f"{track_id}.mp3"
            dest_path = AUDIO_DIR / dest_name
            if not dest_path.exists():
                try:
                    with zf.open(zip_path) as src, open(dest_path, "wb") as dst:
                        dst.write(src.read())
                except KeyError:
                    print(f"WARNING: {zip_path} not found in zip, skipping track {track_id}", flush=True)
                    continue
            manifest_rows.append({
                "track_id": track_id,
                "genre_top": row[("track", "genre_top")],
                "title": row[("track", "title")],
                "artist": row[("artist", "name")],
                "relative_path": f"audio/{dest_name}",
            })
            if (i + 1) % 200 == 0:
                print(f"  extracted {i + 1}/{len(curated)}", flush=True)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(METADATA_DIR / "curated_tracks.csv", index=False)
    print(f"Wrote {len(manifest)} tracks to {METADATA_DIR / 'curated_tracks.csv'}", flush=True)

    print("Cleaning up staged zips ...", flush=True)
    for f in [metadata_zip, small_zip]:
        if os.path.exists(f):
            os.remove(f)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
