import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.prepare_dataset import clip_boundary_box, convert_box, prepare


class DatasetPreparationTests(unittest.TestCase):
    def test_reported_right_edge_error_clips_one_pixel(self):
        values = clip_boundary_box((381, 14, 401, 34), 400, 267)
        self.assertEqual(values, (381, 14, 400, 34))
        line = convert_box("without_mask", *values, 400, 267).split()
        xc, yc, width, height = map(float, line[1:])
        self.assertLessEqual(xc + width / 2, 1.0000001)
        self.assertAlmostEqual(width, 19 / 400)

    def test_each_edge_can_be_clipped_without_shifting_other_edges(self):
        self.assertEqual(clip_boundary_box((-1, -1, 11, 11), 10, 10), (0, 0, 10, 10))
        self.assertEqual(clip_boundary_box((1, 2, 8, 9), 10, 10), (1, 2, 8, 9))

    def test_large_nonfinite_and_degenerate_errors_are_rejected(self):
        for values in ((1, 1, 12, 5), (10, 1, 11, 5), (5, 1, 2, 5),
                       (1, 3, 5, 3), (float("nan"), 1, 5, 5), (1, 1, float("inf"), 5)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                clip_boundary_box(values, 10, 10)

    def test_preparation_logs_correction_and_preserves_raw_xml(self):
        workspace = Path(__file__).resolve().parents[1]
        temporary_root = workspace / "artifacts" / "test_tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            assert Path(temporary).resolve().is_relative_to(workspace)
            raw = Path(temporary) / "raw"
            (raw / "images").mkdir(parents=True)
            (raw / "annotations").mkdir()
            for index in range(3):
                Image.new("RGB", (400, 267), (index * 80, 20, 30)).save(raw / "images" / f"image{index}.png")
                xml = (f"<annotation><filename>image{index}.png</filename>"
                       "<size><width>400</width><height>267</height></size>"
                       "<object><name>without_mask</name><bndbox><xmin>381</xmin>"
                       "<ymin>14</ymin><xmax>401</xmax><ymax>34</ymax>"
                       "</bndbox></object></annotation>")
                (raw / "annotations" / f"image{index}.xml").write_text(xml)
            before = {path.name: path.read_bytes() for path in (raw / "annotations").glob("*.xml")}
            output = Path(temporary) / "yolo"
            prepare(raw, output)
            report = json.loads((output / "summary.json").read_text())
            self.assertEqual(report["corrected_boxes"], 3)
            self.assertEqual(report["box_corrections"][0]["clipped_box"], [381, 14, 400, 34])
            self.assertEqual(len(list((output / "labels").rglob("*.txt"))), 3)
            for path in (raw / "annotations").glob("*.xml"):
                self.assertEqual(before[path.name], path.read_bytes())


if __name__ == "__main__":
    unittest.main()
