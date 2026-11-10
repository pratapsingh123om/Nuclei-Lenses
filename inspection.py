"""Phase One: audit MoNuSeg archives before designing a training notebook.

This script does not extract data, create masks, split patients, or import a
machine-learning framework. It answers four questions:

1. Are the ZIP files intact and internally consistent?
2. Does every TIFF have exactly one matching XML annotation?
3. Are the images and polygon annotations technically valid?
4. Which training pairs belong to the published 30-image MoNuSeg 2018 cohort,
   and which seven are additional pairs in the current 37-image archive?

Outputs are JSON, CSV, and Markdown so the data decision can be reviewed before
any preprocessing or training code is written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from xml.etree import ElementTree as ET

from PIL import Image, UnidentifiedImageError


CANONICAL_TRAIN_ORGANS: dict[str, tuple[str, ...]] = {
    "breast": (
        "TCGA-A7-A13E-01Z-00-DX1",
        "TCGA-A7-A13F-01Z-00-DX1",
        "TCGA-AR-A1AK-01Z-00-DX1",
        "TCGA-AR-A1AS-01Z-00-DX1",
        "TCGA-E2-A1B5-01Z-00-DX1",
        "TCGA-E2-A14V-01Z-00-DX1",
    ),
    "kidney": (
        "TCGA-B0-5711-01Z-00-DX1",
        "TCGA-HE-7128-01Z-00-DX1",
        "TCGA-HE-7129-01Z-00-DX1",
        "TCGA-HE-7130-01Z-00-DX1",
        "TCGA-B0-5710-01Z-00-DX1",
        "TCGA-B0-5698-01Z-00-DX1",
    ),
    "liver": (
        "TCGA-18-5592-01Z-00-DX1",
        "TCGA-38-6178-01Z-00-DX1",
        "TCGA-49-4488-01Z-00-DX1",
        "TCGA-50-5931-01Z-00-DX1",
        "TCGA-21-5784-01Z-00-DX1",
        "TCGA-21-5786-01Z-00-DX1",
    ),
    "prostate": (
        "TCGA-G9-6336-01Z-00-DX1",
        "TCGA-G9-6348-01Z-00-DX1",
        "TCGA-G9-6356-01Z-00-DX1",
        "TCGA-G9-6363-01Z-00-DX1",
        "TCGA-CH-5767-01Z-00-DX1",
        "TCGA-G9-6362-01Z-00-DX1",
    ),
    "bladder": (
        "TCGA-DK-A2I6-01A-01-TS1",
        "TCGA-G2-A2EK-01A-02-TSB",
    ),
    "colon": (
        "TCGA-AY-A8YK-01A-01-TS1",
        "TCGA-NH-A8F7-01A-01-TS1",
    ),
    "stomach": (
        "TCGA-KB-A93J-01A-01-TS1",
        "TCGA-RD-A8N9-01A-01-TS1",
    ),
}

CANONICAL_TRAIN_STEMS = frozenset(
    stem for stems in CANONICAL_TRAIN_ORGANS.values() for stem in stems
)
ORGAN_BY_STEM = {
    stem: organ for organ, stems in CANONICAL_TRAIN_ORGANS.items() for stem in stems
}
RASTER_SUFFIXES = {".tif", ".tiff"}
XML_SUFFIX = ".xml"


def normalize_stem(member: str) -> str:
    stem = PurePosixPath(member).stem
    for suffix in ("_annotations", "_annotation", "_annot"):
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def patient_id(stem: str) -> str:
    parts = stem.split("-")
    if len(parts) >= 3 and parts[0].upper() == "TCGA":
        return "-".join(parts[:3]).upper()
    return stem


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def visible_file_members(handle: zipfile.ZipFile) -> list[str]:
    visible: list[str] = []
    for info in handle.infolist():
        if info.is_dir():
            continue
        parts = PurePosixPath(info.filename).parts
        if "__MACOSX" in parts or any(part.startswith("._") for part in parts):
            continue
        if PurePosixPath(info.filename).name == ".DS_Store":
            continue
        visible.append(info.filename)
    return visible


def sha256_path(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def group_members_by_stem(members: Iterable[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for member in members:
        grouped[normalize_stem(member)].append(member)
    return dict(grouped)


def polygon_area(vertices: list[tuple[float, float]]) -> float:
    total = 0.0
    for index, (x1, y1) in enumerate(vertices):
        x2, y2 = vertices[(index + 1) % len(vertices)]
        total += x1 * y2 - x2 * y1
    return abs(total) * 0.5


def inspect_annotation(
    payload: bytes, width: int, height: int
) -> dict[str, Any]:
    root = ET.fromstring(payload)
    regions = [node for node in root.iter() if local_name(node.tag) == "Region"]
    valid_polygons = 0
    invalid_regions = 0
    invalid_vertices = 0
    out_of_bounds_vertices = 0
    degenerate_polygons = 0
    negative_regions = 0

    for region in regions:
        if str(region.attrib.get("NegativeROA", "0")).lower() in {"1", "true"}:
            negative_regions += 1
        vertices: list[tuple[float, float]] = []
        for node in region.iter():
            if local_name(node.tag) != "Vertex":
                continue
            try:
                x = float(node.attrib["X"])
                y = float(node.attrib["Y"])
            except (KeyError, TypeError, ValueError):
                invalid_vertices += 1
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                invalid_vertices += 1
                continue
            # Coordinates on the outer image boundary are acceptable and will
            # be clipped during later rasterization.
            if not (0.0 <= x <= width and 0.0 <= y <= height):
                out_of_bounds_vertices += 1
            vertices.append((x, y))

        if len(vertices) < 3:
            invalid_regions += 1
        elif polygon_area(vertices) <= 0.5:
            invalid_regions += 1
            degenerate_polygons += 1
        else:
            valid_polygons += 1

    return {
        "region_count": len(regions),
        "valid_polygon_count": valid_polygons,
        "invalid_region_count": invalid_regions,
        "invalid_vertex_count": invalid_vertices,
        "out_of_bounds_vertex_count": out_of_bounds_vertices,
        "degenerate_polygon_count": degenerate_polygons,
        "negative_region_count": negative_regions,
    }


def add_issue(
    issues: list[dict[str, str]],
    severity: str,
    code: str,
    dataset: str,
    detail: str,
    image_id: str = "",
) -> None:
    issues.append(
        {
            "severity": severity,
            "code": code,
            "dataset": dataset,
            "image_id": image_id,
            "detail": detail,
        }
    )


def inspect_archive(
    archive_path: str | Path,
    dataset: str,
    expected_size: tuple[int, int] = (1000, 1000),
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    archive = Path(archive_path)
    records: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    if not archive.exists():
        add_issue(issues, "error", "archive_missing", dataset, str(archive))
        return {"dataset": dataset, "archive": str(archive), "passed": False}, records, issues

    try:
        zip_handle = zipfile.ZipFile(archive)
    except (OSError, zipfile.BadZipFile) as exc:
        add_issue(issues, "error", "invalid_zip", dataset, str(exc))
        return {"dataset": dataset, "archive": str(archive), "passed": False}, records, issues

    with zip_handle:
        bad_member = zip_handle.testzip()
        if bad_member:
            add_issue(issues, "error", "crc_failure", dataset, bad_member)

        members = visible_file_members(zip_handle)
        extension_counts = Counter(
            PurePosixPath(member).suffix.lower() or "<no extension>" for member in members
        )
        tiff_members = [
            member
            for member in members
            if PurePosixPath(member).suffix.lower() in RASTER_SUFFIXES
        ]
        xml_members = [
            member
            for member in members
            if PurePosixPath(member).suffix.lower() == XML_SUFFIX
        ]
        tiff_by_stem = group_members_by_stem(tiff_members)
        xml_by_stem = group_members_by_stem(xml_members)

        for stem, paths in sorted(tiff_by_stem.items()):
            if len(paths) != 1:
                add_issue(
                    issues,
                    "error",
                    "duplicate_image_stem",
                    dataset,
                    f"{len(paths)} TIFF members: {paths}",
                    stem,
                )
        for stem, paths in sorted(xml_by_stem.items()):
            if len(paths) != 1:
                add_issue(
                    issues,
                    "error",
                    "duplicate_xml_stem",
                    dataset,
                    f"{len(paths)} XML members: {paths}",
                    stem,
                )

        all_stems = sorted(set(tiff_by_stem) | set(xml_by_stem))
        content_hashes: dict[str, list[str]] = defaultdict(list)
        for stem in all_stems:
            image_paths = tiff_by_stem.get(stem, [])
            xml_paths = xml_by_stem.get(stem, [])
            if len(image_paths) != 1 or len(xml_paths) != 1:
                if not image_paths:
                    add_issue(issues, "error", "missing_tiff", dataset, "XML has no TIFF", stem)
                if not xml_paths:
                    add_issue(issues, "error", "missing_xml", dataset, "TIFF has no XML", stem)
                continue

            image_member = image_paths[0]
            xml_member = xml_paths[0]
            record: dict[str, Any] = {
                "dataset": dataset,
                "image_id": stem,
                "patient_id": patient_id(stem),
                "image_member": image_member,
                "xml_member": xml_member,
                "cohort": (
                    "canonical_train"
                    if dataset == "train" and stem in CANONICAL_TRAIN_STEMS
                    else "additional_train"
                    if dataset == "train"
                    else "benchmark_test"
                ),
                "organ": ORGAN_BY_STEM.get(stem, "unknown"),
                "valid": True,
            }

            try:
                image_payload = zip_handle.read(image_member)
                record["image_sha256"] = sha256_bytes(image_payload)
                content_hashes[record["image_sha256"]].append(stem)
                with Image.open(io.BytesIO(image_payload)) as image:
                    record["width"], record["height"] = image.size
                    record["mode"] = image.mode
                    record["frames"] = int(getattr(image, "n_frames", 1))
                if (record["width"], record["height"]) != expected_size:
                    record["valid"] = False
                    add_issue(
                        issues,
                        "error",
                        "unexpected_dimensions",
                        dataset,
                        f"Expected {expected_size}; found {(record['width'], record['height'])}",
                        stem,
                    )
                if record["mode"] not in {"RGB", "RGBA"}:
                    record["valid"] = False
                    add_issue(
                        issues,
                        "error",
                        "unexpected_image_mode",
                        dataset,
                        f"Expected RGB/RGBA; found {record['mode']}",
                        stem,
                    )
            except (KeyError, OSError, UnidentifiedImageError) as exc:
                record["valid"] = False
                add_issue(issues, "error", "unreadable_tiff", dataset, str(exc), stem)

            if "width" in record:
                try:
                    xml_payload = zip_handle.read(xml_member)
                    record["xml_sha256"] = sha256_bytes(xml_payload)
                    annotation = inspect_annotation(
                        xml_payload, record["width"], record["height"]
                    )
                    record.update(annotation)
                    if annotation["valid_polygon_count"] == 0:
                        record["valid"] = False
                        add_issue(
                            issues,
                            "error",
                            "no_valid_polygons",
                            dataset,
                            "No valid annotated regions",
                            stem,
                        )
                    # MoNuSeg contours may cross the image boundary by a small
                    # amount. They remain usable after clipping to the image
                    # canvas during rasterization, so these are auditable
                    # warnings rather than archive-integrity failures.
                    for key, code, detail in (
                        (
                            "invalid_region_count",
                            "regions_skipped",
                            "Regions without a usable polygon will be skipped",
                        ),
                        (
                            "invalid_vertex_count",
                            "vertices_skipped",
                            "Non-finite or malformed vertices will be skipped",
                        ),
                        (
                            "out_of_bounds_vertex_count",
                            "vertices_require_clipping",
                            "Finite boundary-crossing vertices will be clipped during mask rasterization",
                        ),
                    ):
                        if annotation[key]:
                            add_issue(
                                issues,
                                "warning",
                                code,
                                dataset,
                                f"{detail}; count: {annotation[key]}",
                                stem,
                            )
                except (KeyError, ET.ParseError, UnicodeError) as exc:
                    record["valid"] = False
                    add_issue(issues, "error", "invalid_xml", dataset, str(exc), stem)
            records.append(record)

        for digest, stems in sorted(content_hashes.items()):
            if len(stems) > 1:
                add_issue(
                    issues,
                    "warning",
                    "duplicate_image_content",
                    dataset,
                    f"Identical image SHA-256 for: {stems}",
                )

    errors = [issue for issue in issues if issue["severity"] == "error"]
    summary = {
        "dataset": dataset,
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256_path(archive),
        "active_member_count": len(members),
        "extension_counts": dict(sorted(extension_counts.items())),
        "directory_counts": dict(
            sorted(Counter(str(PurePosixPath(member).parent) for member in members).items())
        ),
        "tiff_count": len(tiff_members),
        "xml_count": len(xml_members),
        "paired_count": len(records),
        "valid_pair_count": sum(bool(record["valid"]) for record in records),
        "nuclei_total": sum(int(record.get("valid_polygon_count", 0)) for record in records),
        "passed": not errors,
    }
    return summary, records, issues


def evaluate_dataset_policy(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    issues: list[dict[str, str]],
    selected_protocol: str = "canonical_30",
) -> dict[str, Any]:
    if selected_protocol not in {"canonical_30", "extended_37"}:
        raise ValueError("selected_protocol must be 'canonical_30' or 'extended_37'")
    train_by_stem = {record["image_id"]: record for record in train_records}
    test_by_stem = {record["image_id"]: record for record in test_records}
    train_stems = set(train_by_stem)
    test_stems = set(test_by_stem)
    canonical_present = sorted(CANONICAL_TRAIN_STEMS & train_stems)
    canonical_missing = sorted(CANONICAL_TRAIN_STEMS - train_stems)
    additional_train = sorted(train_stems - CANONICAL_TRAIN_STEMS)
    stem_overlap = sorted(train_stems & test_stems)
    patient_overlap = sorted(
        {record["patient_id"] for record in train_records}
        & {record["patient_id"] for record in test_records}
    )

    if canonical_missing:
        add_issue(
            issues,
            "error",
            "canonical_training_ids_missing",
            "cross_dataset",
            str(canonical_missing),
        )
    if len(test_records) != 14:
        add_issue(
            issues,
            "error",
            "unexpected_test_count",
            "test",
            f"Expected 14 paired test cases; found {len(test_records)}",
        )
    if stem_overlap:
        add_issue(
            issues,
            "error",
            "train_test_stem_overlap",
            "cross_dataset",
            str(stem_overlap),
        )
    if patient_overlap:
        add_issue(
            issues,
            "error",
            "train_test_patient_overlap",
            "cross_dataset",
            str(patient_overlap),
        )

    canonical_records = [train_by_stem[stem] for stem in canonical_present]
    canonical_valid = (
        len(canonical_records) == 30
        and all(bool(record["valid"]) for record in canonical_records)
    )
    test_valid = len(test_records) == 14 and all(
        bool(record["valid"]) for record in test_records
    )
    selected_ids = (
        canonical_present if selected_protocol == "canonical_30" else sorted(train_stems)
    )
    return {
        "published_benchmark_training_count": 30,
        "canonical_training_present_count": len(canonical_present),
        "canonical_training_missing": canonical_missing,
        "additional_training_count": len(additional_train),
        "additional_training_ids": additional_train,
        "test_count": len(test_records),
        "train_test_stem_overlap": stem_overlap,
        "train_test_patient_overlap": patient_overlap,
        "benchmark_ready": bool(
            canonical_valid and test_valid and not stem_overlap and not patient_overlap
        ),
        "extended_37_available": bool(
            len(train_records) == 37 and all(bool(record["valid"]) for record in train_records)
        ),
        "selected_protocol": selected_protocol,
        "selected_training_count": len(selected_ids),
        "selected_training_ids": selected_ids,
        "decision_required": False,
        "allowed_decisions": {
            "canonical_30": (
                "Use the published 30-image MoNuSeg cohort, split 24/6 for development, "
                "and reserve the 14-image benchmark test set."
            ),
            "extended_37": (
                "Use all 37 annotated training-archive pairs under a custom protocol; "
                "do not describe this as the original 30-image benchmark training set."
            ),
        },
    }


PAIR_FIELDS = [
    "dataset",
    "cohort",
    "image_id",
    "patient_id",
    "organ",
    "valid",
    "width",
    "height",
    "mode",
    "frames",
    "region_count",
    "valid_polygon_count",
    "invalid_region_count",
    "invalid_vertex_count",
    "out_of_bounds_vertex_count",
    "degenerate_polygon_count",
    "negative_region_count",
    "image_member",
    "xml_member",
    "image_sha256",
    "xml_sha256",
]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(report: dict[str, Any]) -> str:
    train = report["archives"]["train"]
    test = report["archives"]["test"]
    policy = report["policy"]
    errors = report["issue_counts"].get("error", 0)
    warnings = report["issue_counts"].get("warning", 0)
    status = "PASS" if report["data_integrity_passed"] else "FAIL"
    benchmark = "READY" if policy["benchmark_ready"] else "NOT READY"
    extras = ", ".join(policy["additional_training_ids"]) or "None"
    missing = ", ".join(policy["canonical_training_missing"]) or "None"
    return f"""# Phase One — MoNuSeg data inspection

## Verdict

- Data integrity: **{status}**
- Canonical 30 + benchmark 14 protocol: **{benchmark}**
- Errors: {errors}
- Warnings: {warnings}
- Selected protocol: **{policy['selected_protocol']}**
- Selected training images: {policy['selected_training_count']}
- Dataset-policy decision still required: **No**

No preprocessing, splitting, mask generation, or model training was performed.

## Archive inventory

| Archive | Bytes | TIFF | XML | Valid pairs | Annotated nuclei |
|---|---:|---:|---:|---:|---:|
| Training | {train.get('archive_bytes', 0):,} | {train.get('tiff_count', 0)} | {train.get('xml_count', 0)} | {train.get('valid_pair_count', 0)} | {train.get('nuclei_total', 0):,} |
| Test | {test.get('archive_bytes', 0):,} | {test.get('tiff_count', 0)} | {test.get('xml_count', 0)} | {test.get('valid_pair_count', 0)} | {test.get('nuclei_total', 0):,} |

## Benchmark cohort check

- Canonical training IDs present: {policy['canonical_training_present_count']}/30
- Canonical IDs missing: {missing}
- Additional annotated training IDs: {policy['additional_training_count']}
- Additional IDs: {extras}
- Test pairs: {policy['test_count']}/14
- Train/test stem overlap: {policy['train_test_stem_overlap'] or 'None'}
- Train/test patient overlap: {policy['train_test_patient_overlap'] or 'None'}

## Recorded protocol

This inspection run selected **{policy['selected_protocol']}**. Available protocols are:

1. **Canonical 30:** {policy['allowed_decisions']['canonical_30']}
2. **Extended 37:** {policy['allowed_decisions']['extended_37']}

The selected protocol and exact image IDs are stored in `phase1_report.json`.
For the original MoNuSeg benchmark, keep `canonical_30` and reserve all 14 test
images for final evaluation.
"""


def run_inspection(
    train_archive: str | Path,
    test_archive: str | Path,
    output_dir: str | Path,
    expected_size: tuple[int, int] = (1000, 1000),
    selected_protocol: str = "canonical_30",
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_summary, train_records, train_issues = inspect_archive(
        train_archive, "train", expected_size
    )
    test_summary, test_records, test_issues = inspect_archive(
        test_archive, "test", expected_size
    )
    issues = train_issues + test_issues
    policy = evaluate_dataset_policy(
        train_records, test_records, issues, selected_protocol=selected_protocol
    )
    issue_counts = Counter(issue["severity"] for issue in issues)
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "data_inspection_only",
        "data_integrity_passed": issue_counts.get("error", 0) == 0,
        "archives": {"train": train_summary, "test": test_summary},
        "policy": policy,
        "issue_counts": dict(issue_counts),
        "outputs": {
            "pairs_csv": str(output / "phase1_pairs.csv"),
            "issues_csv": str(output / "phase1_issues.csv"),
            "canonical_manifest_csv": str(output / "canonical_30_manifest.csv"),
            "extended_manifest_csv": str(output / "extended_37_manifest.csv"),
            "test_manifest_csv": str(output / "benchmark_14_manifest.csv"),
        },
    }

    all_records = sorted(
        train_records + test_records,
        key=lambda row: (row["dataset"], row["image_id"]),
    )
    write_csv(output / "phase1_pairs.csv", all_records, PAIR_FIELDS)
    write_csv(
        output / "phase1_issues.csv",
        issues,
        ["severity", "code", "dataset", "image_id", "detail"],
    )
    write_csv(
        output / "canonical_30_manifest.csv",
        [row for row in train_records if row["cohort"] == "canonical_train"],
        PAIR_FIELDS,
    )
    write_csv(output / "extended_37_manifest.csv", train_records, PAIR_FIELDS)
    write_csv(output / "benchmark_14_manifest.csv", test_records, PAIR_FIELDS)
    (output / "phase1_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output / "PHASE1_REPORT.md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    return report


def print_summary(report: dict[str, Any]) -> None:
    print("\nPHASE ONE — DATA INSPECTION ONLY")
    print("=" * 38)
    for name in ("train", "test"):
        summary = report["archives"][name]
        print(
            f"{name.title():8} TIFF={summary.get('tiff_count', 0):2} "
            f"XML={summary.get('xml_count', 0):2} "
            f"valid_pairs={summary.get('valid_pair_count', 0):2} "
            f"nuclei={summary.get('nuclei_total', 0):,}"
        )
    print("Data integrity:", "PASS" if report["data_integrity_passed"] else "FAIL")
    print(
        "Canonical benchmark:",
        "READY" if report["policy"]["benchmark_ready"] else "NOT READY",
    )
    print("Additional training IDs:", report["policy"]["additional_training_ids"])
    print("Selected protocol:", report["policy"]["selected_protocol"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-archive",
        default="/content/drive/MyDrive/Nuclei-lens/MoNuSeg 2018 Training Data.zip",
    )
    parser.add_argument(
        "--test-archive",
        default="/content/drive/MyDrive/Nuclei-lens/MoNuSegTestData.zip",
    )
    parser.add_argument(
        "--output-dir",
        default="/content/drive/MyDrive/Nuclei-lens/phase1_inspection",
    )
    parser.add_argument("--expected-width", type=int, default=1000)
    parser.add_argument("--expected-height", type=int, default=1000)
    parser.add_argument(
        "--protocol",
        choices=("canonical_30", "extended_37"),
        default="canonical_30",
        help="Training cohort to record in the inspection report.",
    )
    supplied = list(sys.argv[1:] if argv is None else argv)
    # `%run phase1_inspection.py` and pasted notebook cells inherit Jupyter's
    # private `-f <kernel.json>` argument. Remove only that known pair so real
    # misspelled inspector options still fail loudly.
    cleaned: list[str] = []
    index = 0
    while index < len(supplied):
        if supplied[index] == "-f" and index + 1 < len(supplied):
            index += 2
            continue
        cleaned.append(supplied[index])
        index += 1
    return parser.parse_args(cleaned)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_inspection(
        train_archive=args.train_archive,
        test_archive=args.test_archive,
        output_dir=args.output_dir,
        expected_size=(args.expected_width, args.expected_height),
        selected_protocol=args.protocol,
    )
    print_summary(report)
    print("Reports written to:", args.output_dir)
    return 0 if report["data_integrity_passed"] else 2


if __name__ == "__main__":
    sys.exit(main())


