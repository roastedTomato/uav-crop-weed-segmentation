"""Inspect the downloaded archive and extract original RGB images and masks."""

from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
ARCHIVE = ROOT / "data" / "downloads" / "dataset.zip"
DESTINATION = ROOT / "data" / "raw"
COLORS = {(199, 199, 199), (31, 119, 180), (255, 127, 14)}


def main():
    report = {"archive": ARCHIVE.name, "directories": {}, "originals": []}
    with ARCHIVE.open("rb") as handle:
        report["archive_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    with zipfile.ZipFile(ARCHIVE) as archive:
        bad_file = archive.testzip()
        if bad_file:
            raise ValueError(f"Archive CRC failure: {bad_file}")
        report["crc_check"] = "passed_all_entries"
        names = [n for n in archive.namelist() if not n.endswith("/")]
        report["directories"] = dict(sorted(Counter(str(PurePosixPath(n).parent) for n in names).items()))
        originals = [n for n in names if PurePosixPath(n).parent.name in {"img", "msk"}]
        documents = [n for n in names if PurePosixPath(n).suffix == ".txt"]
        for name in originals + documents:
            relative = PurePosixPath(name).relative_to("Dataset")
            target = DESTINATION.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
        for name in originals:
            content = archive.read(name)
            with Image.open(io.BytesIO(content)) as image:
                image.load()
                entry = {
                    "archive_path": name,
                    "source_id": str(PurePosixPath(name).parent.parent / PurePosixPath(name).stem),
                    "size": list(image.size),
                    "mode": image.mode,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                }
                if PurePosixPath(name).parent.name == "msk":
                    pixels = np.asarray(image.convert("RGB"))
                    known = np.zeros(pixels.shape[:2], dtype=bool)
                    entry["class_pixels"] = {}
                    for color in sorted(COLORS):
                        matches = np.all(pixels == color, axis=-1)
                        entry["class_pixels"][str(color)] = int(matches.sum())
                        known |= matches
                    entry["unknown_pixels"] = int((~known).sum())
                    if entry["unknown_pixels"]:
                        entry["unknown_colors"] = np.unique(pixels[~known], axis=0).tolist()
                report["originals"].append(entry)
        rgb = [e for e in report["originals"] if "/img/" in e["archive_path"]]
        report["duplicate_original_rgb_pixels"] = [h for h, count in Counter(e["pixel_sha256"] for e in rgb).items() if count > 1]
        report["patch_filename_counts"] = {}
        for subset, prefix in [("trainval", "trainval_"), ("test", "test_")]:
            counts = {}
            for original in rgb:
                path = PurePosixPath(original["archive_path"])
                if path.parent.parent.name != subset:
                    continue
                stem = path.stem
                counts[stem] = sum(PurePosixPath(n).name.startswith(stem) for n in names if str(PurePosixPath(n).parent) == f"Dataset/{subset}/img patch")
            report["patch_filename_counts"][subset] = counts
        report["notes"] = [
            "Filename-prefix mapping is a candidate, not a verified patch-to-source mapping.",
            "Original images can be repatched with explicit source IDs; no split is created here.",
            "Distinct hashes exclude exact duplicates only, not spatial overlap or shared flights.",
        ]
    (ARTIFACTS / "DATA_AUDIT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"crc_check": report["crc_check"], "original_files": len(originals), "duplicate_original_rgb_pixels": report["duplicate_original_rgb_pixels"], "patch_filename_counts": report["patch_filename_counts"], "mask_unknown_pixels": {e["archive_path"]: e["unknown_pixels"] for e in report["originals"] if "unknown_pixels" in e}}, indent=2))


if __name__ == "__main__":
    main()
