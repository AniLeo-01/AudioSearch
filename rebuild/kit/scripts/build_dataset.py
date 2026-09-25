#!/usr/bin/env python
"""Download each episode in data/manifest.yaml and cut its two-speaker excerpt (mono, 44.1 kHz, 80 kbps).

Same offsets + same ffmpeg command = same audio timeline, so time-interval labels stay valid.
Usage: python scripts/build_dataset.py [FILE_ID ...]
"""

import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import yaml

DATA = Path("data")

for e in yaml.safe_load((DATA / "manifest.yaml").read_text())["files"]:
    out = DATA / "audio" / f"{e['file_id']}.mp3"
    if out.exists() or (sys.argv[1:] and e["file_id"] not in sys.argv[1:]):
        continue
    out.parent.mkdir(parents=True, exist_ok=True)
    start, dur = e["excerpt"]["start"], e["excerpt"]["end"] - e["excerpt"]["start"]
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "episode.mp3"
        req = urllib.request.Request(e["source_audio_url"], headers={"User-Agent": "audiosearch-dataset/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r, src.open("wb") as f:
            shutil.copyfileobj(r, f)
        fade = f"afade=t=in:d=0.08,afade=t=out:st={dur - 0.3:.2f}:d=0.3"
        cmd = ["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", f"{dur:.2f}", "-i", str(src),
               "-ac", "1", "-ar", "44100", "-b:a", "80k", "-af", fade, "-map_metadata", "-1", str(out)]  # fmt: skip
        subprocess.run(cmd, check=True)
    print("wrote", out)
