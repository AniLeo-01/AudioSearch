"""The problem statement's dataset contract: 5-6 files, 8-10 minutes each, a unique pair of speakers per file."""

import subprocess
from pathlib import Path

import yaml

from audiosearch import config


def test_dataset_contract():
    files = yaml.safe_load((config.DATA / "manifest.yaml").read_text())["files"]
    assert 5 <= len(files) <= 6
    names = [n for f in files for n in (f["speakers"]["host"], f["speakers"]["guest"])]
    assert len(set(names)) == len(names)  # nobody appears in two files, so every pair is unique
    for f in files:
        path = Path(config.DATA / "audio" / f"{f['file_id']}.mp3")
        probe = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]
        assert 480 <= float(subprocess.run(probe, capture_output=True, text=True, check=True).stdout) <= 600
