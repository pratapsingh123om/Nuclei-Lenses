# Copy this entire cell into Colab and run it. No upload or %run is needed.
from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path, PurePosixPath
from collections import Counter, defaultdict
from xml.etree import ElementTree as ET
from datetime import datetime, timezone
from PIL import Image
import csv, hashlib, io, json, math, zipfile

TRAIN_ZIP = Path("/content/drive/MyDrive/Nuclei-lens/MoNuSeg 2018 Training Data.zip")
TEST_ZIP = Path("/content/drive/MyDrive/Nuclei-lens/MoNuSegTestData.zip")
OUT = Path("/content/drive/MyDrive/Nuclei-lens/phase1_inspection")
OUT.mkdir(parents=True, exist_ok=True)
SELECTED_PROTOCOL = "canonical_30"  # Original MoNuSeg benchmark: 24 train, 6 validation.

ORGANS = {
    "breast": ["TCGA-A7-A13E-01Z-00-DX1", "TCGA-A7-A13F-01Z-00-DX1", "TCGA-AR-A1AK-01Z-00-DX1", "TCGA-AR-A1AS-01Z-00-DX1", "TCGA-E2-A1B5-01Z-00-DX1", "TCGA-E2-A14V-01Z-00-DX1"],
    "kidney": ["TCGA-B0-5711-01Z-00-DX1", "TCGA-HE-7128-01Z-00-DX1", "TCGA-HE-7129-01Z-00-DX1", "TCGA-HE-7130-01Z-00-DX1", "TCGA-B0-5710-01Z-00-DX1", "TCGA-B0-5698-01Z-00-DX1"],
    "liver": ["TCGA-18-5592-01Z-00-DX1", "TCGA-38-6178-01Z-00-DX1", "TCGA-49-4488-01Z-00-DX1", "TCGA-50-5931-01Z-00-DX1", "TCGA-21-5784-01Z-00-DX1", "TCGA-21-5786-01Z-00-DX1"],
    "prostate": ["TCGA-G9-6336-01Z-00-DX1", "TCGA-G9-6348-01Z-00-DX1", "TCGA-G9-6356-01Z-00-DX1", "TCGA-G9-6363-01Z-00-DX1", "TCGA-CH-5767-01Z-00-DX1", "TCGA-G9-6362-01Z-00-DX1"],
    "bladder": ["TCGA-DK-A2I6-01A-01-TS1", "TCGA-G2-A2EK-01A-02-TSB"],
    "colon": ["TCGA-AY-A8YK-01A-01-TS1", "TCGA-NH-A8F7-01A-01-TS1"],
    "stomach": ["TCGA-KB-A93J-01A-01-TS1", "TCGA-RD-A8N9-01A-01-TS1"],
}
CANONICAL = {stem for stems in ORGANS.values() for stem in stems}
ORGAN = {stem: organ for organ, stems in ORGANS.items() for stem in stems}

def stem(name):
    value = PurePosixPath(name).stem
    for suffix in ("_annotations", "_annotation", "_annot"):
        if value.lower().endswith(suffix):
            value = value[:-len(suffix)]
    return value

def patient(image_id):
    parts = image_id.split("-")
    return "-".join(parts[:3]).upper() if len(parts) >= 3 and parts[0].upper() == "TCGA" else image_id

def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def tag_name(tag):
    return tag.rsplit("}", 1)[-1]

def area(points):
    return abs(sum(
        x1 * points[(index + 1) % len(points)][1] - points[(index + 1) % len(points)][0] * y1
        for index, (x1, y1) in enumerate(points)
    )) / 2

def inspect_xml(payload, width, height):
    root = ET.fromstring(payload)
    regions = [node for node in root.iter() if tag_name(node.tag) == "Region"]
    result = {"regions": len(regions), "valid_polygons": 0, "invalid_regions": 0,
              "invalid_vertices": 0, "out_of_bounds_vertices": 0}
    for region in regions:
        points = []
        for node in region.iter():
            if tag_name(node.tag) != "Vertex":
                continue
            try:
                x, y = float(node.attrib["X"]), float(node.attrib["Y"])
                if not (math.isfinite(x) and math.isfinite(y)):
                    raise ValueError
                if not (0 <= x <= width and 0 <= y <= height):
                    result["out_of_bounds_vertices"] += 1
                points.append((x, y))
            except (KeyError, TypeError, ValueError):
                result["invalid_vertices"] += 1
        if len(points) >= 3 and area(points) > 0.5:
            result["valid_polygons"] += 1
        else:
            result["invalid_regions"] += 1
    return result

def add_issue(issues, severity, code, split, detail, image_id=""):
    issues.append({"severity": severity, "code": code, "split": split,
                   "image_id": image_id, "detail": detail})

def inspect_zip(path, split):
    records, issues = [], []
    print(f"\nInspecting {split} archive: {path.name}")
    if not path.exists():
        add_issue(issues, "ERROR", "archive_missing", split, str(path))
        return {}, records, issues
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        add_issue(issues, "ERROR", "invalid_zip", split, str(exc))
        return {}, records, issues
    with archive:
        bad = archive.testzip()
        if bad:
            add_issue(issues, "ERROR", "crc_failure", split, bad)
        members = []
        for info in archive.infolist():
            parts = PurePosixPath(info.filename).parts
            if info.is_dir() or "__MACOSX" in parts or any(p.startswith("._") for p in parts) or PurePosixPath(info.filename).name == ".DS_Store":
                continue
            members.append(info.filename)
        ext = Counter(PurePosixPath(name).suffix.lower() or "<no extension>" for name in members)
        images = [name for name in members if PurePosixPath(name).suffix.lower() in {".tif", ".tiff"}]
        xmls = [name for name in members if PurePosixPath(name).suffix.lower() == ".xml"]
        image_map, xml_map = defaultdict(list), defaultdict(list)
        for name in images: image_map[stem(name)].append(name)
        for name in xmls: xml_map[stem(name)].append(name)
        all_ids = sorted(set(image_map) | set(xml_map))
        for item_number, image_id in enumerate(all_ids, start=1):
            if item_number == 1 or item_number % 10 == 0 or item_number == len(all_ids):
                print(f"  checking pair {item_number}/{len(all_ids)}")
            if len(image_map[image_id]) != 1 or len(xml_map[image_id]) != 1:
                if len(image_map[image_id]) != 1:
                    add_issue(issues, "ERROR", "image_pairing", split, str(image_map[image_id]), image_id)
                if len(xml_map[image_id]) != 1:
                    add_issue(issues, "ERROR", "xml_pairing", split, str(xml_map[image_id]), image_id)
                continue
            record = {"split": split, "image_id": image_id, "patient_id": patient(image_id),
                      "cohort": "canonical_train" if split == "train" and image_id in CANONICAL else "additional_train" if split == "train" else "benchmark_test",
                      "organ": ORGAN.get(image_id, "unknown"), "image_member": image_map[image_id][0],
                      "xml_member": xml_map[image_id][0], "valid": True}
            try:
                payload = archive.read(record["image_member"])
                with Image.open(io.BytesIO(payload)) as image:
                    record.update(width=image.width, height=image.height, mode=image.mode)
                if (record["width"], record["height"]) != (1000, 1000):
                    record["valid"] = False
                    add_issue(issues, "ERROR", "dimensions", split, f"{record['width']}x{record['height']}", image_id)
                if record["mode"] not in {"RGB", "RGBA"}:
                    record["valid"] = False
                    add_issue(issues, "ERROR", "image_mode", split, record["mode"], image_id)
            except Exception as exc:
                record["valid"] = False
                add_issue(issues, "ERROR", "unreadable_tiff", split, str(exc), image_id)
            try:
                annotation = inspect_xml(archive.read(record["xml_member"]), record.get("width", 1000), record.get("height", 1000))
                record.update(annotation)
                if annotation["valid_polygons"] == 0:
                    record["valid"] = False
                    add_issue(issues, "ERROR", "no_valid_polygons", split, str(annotation), image_id)
                if annotation["invalid_regions"]:
                    add_issue(issues, "WARNING", "regions_skipped", split,
                              f"Unusable regions skipped: {annotation['invalid_regions']}", image_id)
                if annotation["invalid_vertices"]:
                    add_issue(issues, "WARNING", "vertices_skipped", split,
                              f"Malformed/non-finite vertices skipped: {annotation['invalid_vertices']}", image_id)
                if annotation["out_of_bounds_vertices"]:
                    add_issue(issues, "WARNING", "vertices_require_clipping", split,
                              f"Finite vertices to clip at rasterization: {annotation['out_of_bounds_vertices']}", image_id)
            except Exception as exc:
                record["valid"] = False
                add_issue(issues, "ERROR", "invalid_xml", split, str(exc), image_id)
            records.append(record)
    summary = {"archive": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path),
               "active_files": len(members), "extensions": dict(ext), "tif": len(images), "xml": len(xmls),
               "pairs": len(records), "valid_pairs": sum(r["valid"] for r in records),
               "nuclei": sum(r.get("valid_polygons", 0) for r in records)}
    return summary, records, issues

train_summary, train_records, train_issues = inspect_zip(TRAIN_ZIP, "train")
test_summary, test_records, test_issues = inspect_zip(TEST_ZIP, "test")
issues = train_issues + test_issues
train_ids, test_ids = {r["image_id"] for r in train_records}, {r["image_id"] for r in test_records}
missing_canonical = sorted(CANONICAL - train_ids)
extras = sorted(train_ids - CANONICAL)
stem_overlap = sorted(train_ids & test_ids)
patient_overlap = sorted({r["patient_id"] for r in train_records} & {r["patient_id"] for r in test_records})
if missing_canonical: add_issue(issues, "ERROR", "canonical_ids_missing", "cross", str(missing_canonical))
if len(test_records) != 14: add_issue(issues, "ERROR", "test_count", "test", f"Found {len(test_records)}, expected 14")
if stem_overlap: add_issue(issues, "ERROR", "stem_overlap", "cross", str(stem_overlap))
if patient_overlap: add_issue(issues, "ERROR", "patient_overlap", "cross", str(patient_overlap))
errors = [item for item in issues if item["severity"] == "ERROR"]
benchmark_ready = not missing_canonical and len(test_records) == 14 and not stem_overlap and not patient_overlap and all(r["valid"] for r in train_records if r["image_id"] in CANONICAL) and all(r["valid"] for r in test_records)

report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "phase": "data_inspection_only",
          "integrity_pass": not errors, "benchmark_ready": benchmark_ready,
          "archives": {"train": train_summary, "test": test_summary},
          "canonical_present": len(CANONICAL & train_ids), "canonical_missing": missing_canonical,
          "additional_training_ids": extras, "train_test_stem_overlap": stem_overlap,
          "train_test_patient_overlap": patient_overlap, "issues": issues,
          "selected_protocol": SELECTED_PROTOCOL,
          "selected_training_ids": sorted(CANONICAL if SELECTED_PROTOCOL == "canonical_30" else train_ids),
          "decision_required": False}

fields = ["split", "cohort", "image_id", "patient_id", "organ", "valid", "width", "height", "mode",
          "regions", "valid_polygons", "invalid_regions", "invalid_vertices", "out_of_bounds_vertices", "image_member", "xml_member"]
def write_csv(name, rows, columns):
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
write_csv("phase1_pairs.csv", train_records + test_records, fields)
write_csv("phase1_issues.csv", issues, ["severity", "code", "split", "image_id", "detail"])
write_csv("canonical_30_manifest.csv", [r for r in train_records if r["cohort"] == "canonical_train"], fields)
write_csv("extended_37_manifest.csv", train_records, fields)
write_csv("benchmark_14_manifest.csv", test_records, fields)
(OUT / "phase1_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

status = "PASS" if not errors else "FAIL"
ready = "READY" if benchmark_ready else "NOT READY"
markdown = f"""# Phase One — MoNuSeg data inspection

- Data integrity: **{status}**
- Canonical 30 + benchmark 14: **{ready}**
- Training pairs: {len(train_records)}
- Test pairs: {len(test_records)}
- Canonical IDs present: {len(CANONICAL & train_ids)}/30
- Additional training IDs ({len(extras)}): {', '.join(extras) or 'None'}
- Errors: {len(errors)}
- Warnings: {len([item for item in issues if item['severity'] == 'WARNING'])}
- Selected protocol: **{SELECTED_PROTOCOL}**

## Recorded protocol

`canonical_30` is selected for the original benchmark: split these 30 images
into 24 training and 6 validation images, and reserve all 14 test images.
Finite out-of-bounds vertices are clip-safe and will be clipped to the image
canvas when masks are rasterized.
No extraction, masks, patches, split, or training was performed.
"""
(OUT / "PHASE1_REPORT.md").write_text(markdown, encoding="utf-8")
print(markdown)
print("Reports saved to:", OUT)
if errors:
    print("\nFirst issues:")
    for item in errors[:10]: print(item)

