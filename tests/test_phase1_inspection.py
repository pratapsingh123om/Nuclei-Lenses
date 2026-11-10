from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from inspection import CANONICAL_TRAIN_STEMS, parse_args, run_inspection


def image_bytes(size: tuple[int, int] = (16, 16)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(165, 92, 132)).save(buffer, format="TIFF")
    return buffer.getvalue()


def annotation_bytes() -> bytes:
    return b"""<Annotations><Annotation><Regions><Region><Vertices>
    <Vertex X="2" Y="2"/><Vertex X="12" Y="2"/><Vertex X="7" Y="12"/>
    </Vertices></Region></Regions></Annotation></Annotations>"""


def boundary_annotation_bytes() -> bytes:
    return b"""<Annotations><Annotation><Regions>
    <Region><Vertices><Vertex X="-0.25" Y="2"/><Vertex X="16.25" Y="2"/>
    <Vertex X="8" Y="14"/></Vertices></Region>
    <Region><Vertices><Vertex X="4" Y="4"/><Vertex X="4" Y="4"/></Vertices></Region>
    </Regions></Annotation></Annotations>"""


def write_archive(
    path: Path,
    stems: list[str],
    flat: bool,
    annotation_payload: bytes | None = None,
) -> None:
    annotation_payload = annotation_payload or annotation_bytes()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for stem in stems:
            image_name = f"{stem}.tif" if flat else f"Dataset/Tissue Images/{stem}.tif"
            xml_name = f"{stem}.xml" if flat else f"Dataset/Annotations/{stem}.xml"
            handle.writestr(image_name, image_bytes())
            handle.writestr(xml_name, annotation_payload)


class PhaseOneInspectionTests(unittest.TestCase):
    def test_colab_kernel_argument_is_ignored(self) -> None:
        args = parse_args(
            [
                "-f",
                "/root/.local/share/jupyter/runtime/kernel-example.json",
                "--output-dir",
                "/tmp/report",
            ]
        )
        self.assertEqual(args.output_dir, "/tmp/report")

    def test_canonical_30_plus_seven_and_test_14(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.zip"
            test = root / "test.zip"
            extras = [f"TCGA-ZZ-EX{index:02d}-01Z-00-DX1" for index in range(7)]
            test_ids = [f"TCGA-TT-{index:04d}-01A-01-TS1" for index in range(14)]
            write_archive(train, sorted(CANONICAL_TRAIN_STEMS) + extras, flat=False)
            write_archive(test, test_ids, flat=True)

            report = run_inspection(train, test, root / "report", expected_size=(16, 16))

            self.assertTrue(report["data_integrity_passed"])
            self.assertTrue(report["policy"]["benchmark_ready"])
            self.assertTrue(report["policy"]["extended_37_available"])
            self.assertEqual(report["policy"]["selected_protocol"], "canonical_30")
            self.assertFalse(report["policy"]["decision_required"])
            self.assertEqual(report["policy"]["additional_training_count"], 7)
            self.assertTrue((root / "report" / "canonical_30_manifest.csv").exists())

    def test_boundary_crossing_vertices_are_clip_safe_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.zip"
            test = root / "test.zip"
            write_archive(
                train,
                sorted(CANONICAL_TRAIN_STEMS),
                flat=False,
                annotation_payload=boundary_annotation_bytes(),
            )
            write_archive(
                test,
                [f"TEST-{index}" for index in range(14)],
                flat=True,
                annotation_payload=boundary_annotation_bytes(),
            )

            report = run_inspection(train, test, root / "report", expected_size=(16, 16))

            self.assertTrue(report["data_integrity_passed"])
            self.assertTrue(report["policy"]["benchmark_ready"])
            self.assertGreater(report["issue_counts"].get("warning", 0), 0)
            issues = (root / "report" / "phase1_issues.csv").read_text(encoding="utf-8")
            self.assertIn("vertices_require_clipping", issues)
            self.assertIn("regions_skipped", issues)

    def test_missing_xml_fails_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.zip"
            test = root / "test.zip"
            write_archive(train, sorted(CANONICAL_TRAIN_STEMS), flat=False)
            write_archive(test, [f"TEST-{index}" for index in range(14)], flat=True)
            with zipfile.ZipFile(train, "a") as handle:
                handle.writestr("Dataset/Tissue Images/ORPHAN.tif", image_bytes())

            report = run_inspection(train, test, root / "report", expected_size=(16, 16))

            self.assertFalse(report["data_integrity_passed"])
            issues = (root / "report" / "phase1_issues.csv").read_text(encoding="utf-8")
            self.assertIn("missing_xml", issues)


if __name__ == "__main__":
    unittest.main()


