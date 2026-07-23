"""One-time enrichment pass: recovers genres_all, album id/title, and
track-level free-text tags for already-curated songs -- fields
scripts/acquire_fma.py trims away during its very first parse of FMA's
tracks.csv (see that script's docstring), before either the raw file or its
own trimmed curated_tracks.csv survive on disk (both are gitignored and the
staged zip is deleted at the end of that run). Re-downloads the same
fma_metadata.zip acquire_fma.py used, since there's nothing left locally to
re-parse instead.

Matches rows purely by fma_track_id (the DB's own idempotency key), so this
is safe to re-run and only ever updates songs already present in --db; it
never adds or removes songs. Run once against the real local library
(data/artifacts/sonic_explorer.db); deploy_data/ picks up the new fields via
scripts/build_deploy_subset.py's normal rebuild-from-data/ pass, not by
running this script a second time against deploy_data directly.
"""

import argparse
import ast
import json
import os
import sys
import zipfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sonic_explorer.config import PROJECT_ROOT  # noqa: E402
from sonic_explorer.repository.db import init_db  # noqa: E402
from sonic_explorer.repository.song_repository import SongRepository  # noqa: E402

FMA_METADATA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
STAGING_DIR = PROJECT_ROOT / "data" / "_downloads"


def download(url: str, dest: str) -> None:
    """Same streamed-download-with-progress logic as acquire_fma.py's
    download() -- duplicated rather than imported since scripts/ isn't a
    package (no __init__.py), so cross-script imports aren't set up here."""
    import requests

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


def _parse_list_field(raw) -> list:
    """FMA's tracks.csv stores list-valued fields (genres_all, tags) as
    Python-repr strings, e.g. "[15, 12]" or "['ambient', 'chill']" --
    ast.literal_eval, not json.loads, since it's Python repr, not JSON."""
    if pd.isna(raw):
        return []
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    return list(parsed) if isinstance(parsed, (list, tuple)) else []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="Path to the sonic_explorer.db to enrich")
    args = parser.parse_args()

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    metadata_zip = str(STAGING_DIR / "fma_metadata.zip")
    download(FMA_METADATA_URL, metadata_zip)

    print("Parsing tracks.csv ...", flush=True)
    with zipfile.ZipFile(metadata_zip) as zf:
        zf.extract("fma_metadata/tracks.csv", str(STAGING_DIR))
    tracks = pd.read_csv(STAGING_DIR / "fma_metadata" / "tracks.csv", index_col=0, header=[0, 1])

    conn = init_db(args.db)
    song_repo = SongRepository(conn)
    songs = song_repo.list_songs()
    print(f"Enriching {len(songs)} songs in {args.db} ...", flush=True)

    updated, missing = 0, 0
    for song in songs:
        if song.fma_track_id not in tracks.index:
            print(f"WARNING: fma_track_id {song.fma_track_id} not found in tracks.csv, skipping", flush=True)
            missing += 1
            continue
        row = tracks.loc[song.fma_track_id]

        genres_all = _parse_list_field(row[("track", "genres_all")])
        track_tags = _parse_list_field(row[("track", "tags")])
        raw_album_id = row[("album", "id")]
        album_id = int(raw_album_id) if pd.notna(raw_album_id) else None
        raw_album_title = row[("album", "title")]
        album_title = raw_album_title if pd.notna(raw_album_title) else None

        song_repo.update_metadata_extras(
            song.id,
            genres_all=json.dumps(genres_all),
            album_id=album_id,
            album_title=album_title,
            track_tags=json.dumps(track_tags),
        )
        updated += 1

    print(f"Updated {updated}/{len(songs)} songs ({missing} not found in tracks.csv).", flush=True)

    print("Cleaning up staged zip ...", flush=True)
    if os.path.exists(metadata_zip):
        os.remove(metadata_zip)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
