"""Create reproducible image-grouped splits and RGB/label patches."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
RAW = ROOT / "data" / "raw"
PATCHES = ROOT / "data" / "patches"
SIZE = 256
SEED = 42
IGNORE = 255
COLORS = [(199, 199, 199), (31, 119, 180), (255, 127, 14)]


def source_records():
    candidates = sorted((RAW / "trainval" / "img").glob("*.jpg"))
    if len(candidates) != 12:
        raise ValueError("Expected 12 trainval original images. Run the data audit first.")
    # Choose validation sources before creating any patches.
    indices = np.random.default_rng(SEED).permutation(len(candidates))[:3]
    validation = {candidates[int(i)].stem for i in indices}
    groups = [(p, "val" if p.stem in validation else "train") for p in candidates]
    tests = sorted((RAW / "test" / "img").glob("*.jpg"))
    if len(tests) != 7:
        raise ValueError("Expected seven original test images.")
    groups += [(p, "test") for p in tests]
    for stage in (15, 19):
        path = RAW / "test_different_bbch" / f"BBCH{stage}" / "img" / f"bbch{stage}_img.jpg"
        groups.append((path, f"extra_bbch{stage}"))
    records = []
    for image, split in groups:
        mask = image.parent.parent / "msk" / (image.stem.replace("_img", "_msk") + ".png")
        if not image.exists() or not mask.exists():
            raise FileNotFoundError(f"Missing image/mask pair: {image}, {mask}")
        source = image.relative_to(RAW).with_suffix("").as_posix()
        records.append({"source_id": source, "split": split,
                        "image_path": image.relative_to(ROOT).as_posix(),
                        "mask_path": mask.relative_to(ROOT).as_posix()})
    return records


def encode_mask(rgb):
    labels = np.full(rgb.shape[:2], IGNORE, dtype=np.uint8)
    for class_id, color in enumerate(COLORS):
        labels[np.all(rgb == color, axis=-1)] = class_id
    if np.any(labels == IGNORE):
        raise ValueError("Original mask contains an unknown RGB color.")
    return labels


def verify_groups(records):
    groups = {}
    for record in records:
        groups.setdefault(record["split"], set()).add(record["source_id"])
    intersections = {}
    keys = sorted(groups)
    for index, left in enumerate(keys):
        for right in keys[index + 1:]:
            common = groups[left] & groups[right]
            if common:
                raise ValueError(f"Source overlap: {left}, {right}, {common}")
            intersections[f"{left} / {right}"] = []
    return intersections


def main():
    records = source_records()
    intersections = verify_groups(records)
    rows = []
    original_hashes = {}
    class_counts = {}
    for record in records:
        with Image.open(ROOT / record["image_path"]) as image:
            rgb = np.asarray(image.convert("RGB"))
        with Image.open(ROOT / record["mask_path"]) as mask:
            labels = encode_mask(np.asarray(mask.convert("RGB")))
        if rgb.shape[:2] != labels.shape:
            raise ValueError("Original image/mask dimensions differ.")
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        if digest in original_hashes:
            raise ValueError("Exact duplicate original RGB images detected.")
        original_hashes[digest] = record["source_id"]
        record["rgb_pixel_sha256"] = digest
        height, width = labels.shape
        record.update(width=width, height=height)
        split = record["split"]
        class_counts.setdefault(split, np.zeros(3, dtype=np.int64))
        source_name = record["source_id"].replace("/", "__")
        for y in range(0, height, SIZE):
            for x in range(0, width, SIZE):
                valid_h, valid_w = min(SIZE, height - y), min(SIZE, width - x)
                crop = rgb[y:y + valid_h, x:x + valid_w]
                target = labels[y:y + valid_h, x:x + valid_w]
                patch = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
                patch[:valid_h, :valid_w] = crop
                label_patch = np.full((SIZE, SIZE), IGNORE, dtype=np.uint8)
                label_patch[:valid_h, :valid_w] = target
                name = f"{source_name}__y{y:04d}_x{x:04d}.png"
                image_path = PATCHES / split / "img" / name
                mask_path = PATCHES / split / "msk" / name
                image_path.parent.mkdir(parents=True, exist_ok=True)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(patch).save(image_path)
                Image.fromarray(label_patch).save(mask_path)
                # Verify saved pixels, including sparse labels and ignored padding.
                with Image.open(image_path) as saved:
                    if not np.array_equal(np.asarray(saved), patch):
                        raise ValueError("Saved RGB patch differs from source crop.")
                with Image.open(mask_path) as saved:
                    if not np.array_equal(np.asarray(saved), label_patch):
                        raise ValueError("Saved label patch differs from source crop.")
                counts = np.bincount(target.ravel(), minlength=3)
                class_counts[split] += counts
                rows.append({"source_id": record["source_id"], "split": split,
                             "image_path": image_path.relative_to(ROOT).as_posix(),
                             "mask_path": mask_path.relative_to(ROOT).as_posix(),
                             "x": x, "y": y, "valid_width": valid_w, "valid_height": valid_h,
                             "class_pixels": counts.tolist(),
                             "ignored_pixels": SIZE * SIZE - valid_w * valid_h})
        print(f"Prepared {record['source_id']} -> {split}", flush=True)
    verify_groups(rows)
    if len({r["image_path"] for r in rows}) != len(rows):
        raise ValueError("Duplicate patch paths.")
    if sum(r["valid_width"] * r["valid_height"] for r in rows) != sum(r["width"] * r["height"] for r in records):
        raise ValueError("Patch coverage does not match original pixels.")
    summary = {"seed": SEED, "patch_size": SIZE, "ignore_label": IGNORE,
               "class_colors": COLORS, "source_counts": dict(Counter(r["split"] for r in records)),
               "patch_counts": dict(Counter(r["split"] for r in rows)),
               "class_pixels": {k: v.tolist() for k, v in class_counts.items()},
               "source_intersections": intersections, "all_saved_patches_verified": True,
               "coverage_verified": True, "exact_duplicate_originals": 0,
               "limitation": "Flight identity and spatial overlap across different originals remain unverified."}
    (ARTIFACTS / "SOURCE_SPLITS.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    with (ARTIFACTS / "PATCH_MANIFEST.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (ARTIFACTS / "SPLIT_CHECKS.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
