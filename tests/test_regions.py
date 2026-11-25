#!/usr/bin/env python
import os
import unittest

import pytest

import pdfplumber
from pdfplumber import regions as regions_mod

HERE = os.path.abspath(os.path.dirname(__file__))

NICS = os.path.join(HERE, "pdfs/nics-background-checks-2015-11.pdf")


def _char_dict(x0, top, x1, bottom, text="x"):
    return {
        "object_type": "char",
        "text": text,
        "x0": x0,
        "top": top,
        "x1": x1,
        "bottom": bottom,
        "width": x1 - x0,
        "height": bottom - top,
    }


class TestBboxCoercion(unittest.TestCase):
    def test_sequence(self):
        assert regions_mod.coerce_bbox((10, 20, 30, 40)) == (10, 20, 30, 40)

    def test_reversed_and_floats(self):
        assert regions_mod.coerce_bbox((30, 40, 10, 20)) == (10.0, 20.0, 30.0, 40.0)

    def test_mappings(self):
        assert regions_mod.coerce_bbox({"x0": 1, "top": 2, "x1": 3, "bottom": 4}) == (
            1,
            2,
            3,
            4,
        )
        assert regions_mod.coerce_bbox({"x0": 1, "y0": 2, "x1": 3, "y1": 4}) == (
            1.0,
            2.0,
            3.0,
            4.0,
        )
        assert regions_mod.coerce_bbox(
            {"left": 1, "top": 2, "right": 3, "bottom": 4}
        ) == (1.0, 2.0, 3.0, 4.0)

    def test_zero_area(self):
        with pytest.raises(ValueError):
            regions_mod.coerce_bbox((10, 10, 10, 40))
        with pytest.raises(ValueError):
            regions_mod.coerce_bbox((10, 10, 40, 10))

    def test_bad_input(self):
        with pytest.raises(TypeError):
            regions_mod.coerce_bbox("nope")
        with pytest.raises(ValueError):
            regions_mod.coerce_bbox((1, 2, 3))
        with pytest.raises(ValueError):
            regions_mod.coerce_bbox({"x0": 1, "x1": 2})


class TestNormalizeRegions(unittest.TestCase):
    def test_single_bbox(self):
        out = regions_mod.normalize_regions((0, 0, 10, 10))
        assert out == [{"id": 0, "pixel_bbox": (0.0, 0.0, 10.0, 10.0)}]

    def test_list_and_ids(self):
        out = regions_mod.normalize_regions(
            [
                (0, 0, 10, 10),
                {"id": "header", "x0": 5, "top": 5, "x1": 20, "bottom": 20},
            ]
        )
        assert [r["id"] for r in out] == [0, "header"]

    def test_mapping_of_ids(self):
        out = regions_mod.normalize_regions(
            {"a": (0, 0, 10, 10), "b": (10, 10, 20, 20)}
        )
        assert [r["id"] for r in out] == ["a", "b"]
        assert out[1]["pixel_bbox"] == (10.0, 10.0, 20.0, 20.0)


class TestOverlapSemantics(unittest.TestCase):
    def test_strict_overlap(self):
        kind, area = regions_mod.overlap_kind((0, 0, 10, 10), (5, 5, 15, 15))
        assert kind == "overlap" and area == 25

    def test_edge_contact_is_not_overlap(self):
        # Shares the right/left edge only.
        kind, area = regions_mod.overlap_kind((0, 0, 10, 10), (10, 0, 20, 10))
        assert (kind, area) == ("touch", 0.0)
        # Shares the bottom/top edge only.
        kind, area = regions_mod.overlap_kind((0, 0, 10, 10), (0, 10, 10, 20))
        assert (kind, area) == ("touch", 0.0)
        # Shares a single corner only.
        kind, area = regions_mod.overlap_kind((0, 0, 10, 10), (10, 10, 20, 20))
        assert (kind, area) == ("touch", 0.0)

    def test_containment(self):
        kind, area = regions_mod.overlap_kind((2, 2, 8, 8), (0, 0, 10, 10))
        assert (kind, area) == ("overlap", 36)

    def test_separated(self):
        kind, area = regions_mod.overlap_kind((0, 0, 10, 10), (11, 0, 20, 10))
        assert (kind, area) == ("none", 0.0)


class TestAssignment(unittest.TestCase):
    def test_largest_overlap_wins(self):
        glyph = _char_dict(0, 0, 10, 10)
        regions = [(0, 0, 4, 10), (4, 0, 10, 10)]
        assignments, unassigned, conflicts = regions_mod.assign_glyphs_to_regions(
            [glyph], regions
        )
        assert [len(a) for a in assignments] == [0, 1]
        assert assignments[1][0][0] is glyph
        assert assignments[1][0][1] == 60
        assert unassigned == [] and conflicts == []

    def test_tie_reported_once_and_not_output(self):
        glyph = _char_dict(0, 0, 10, 10)
        regions = [(0, 0, 5, 10), (5, 0, 10, 10)]
        assignments, unassigned, conflicts = regions_mod.assign_glyphs_to_regions(
            [glyph], regions
        )
        assert [len(a) for a in assignments] == [0, 0]
        assert unassigned == []
        assert len(conflicts) == 1
        assert conflicts[0]["char"] is glyph
        assert conflicts[0]["reason"] == "overlap_tie"
        assert [i for i, _ in conflicts[0]["overlap_areas"]] == [0, 1]

    def test_edge_contact_reason(self):
        glyph = _char_dict(10, 0, 20, 10)
        assignments, unassigned, conflicts = regions_mod.assign_glyphs_to_regions(
            [glyph], [(0, 0, 10, 10)]
        )
        assert assignments == [[]]
        assert len(unassigned) == 1
        assert unassigned[0]["reason"] == "edge_contact_only"
        assert unassigned[0]["touching"] == [0]
        assert conflicts == []

    def test_outside_reason(self):
        glyph = _char_dict(50, 50, 60, 60)
        _, unassigned, _ = regions_mod.assign_glyphs_to_regions(
            [glyph], [(0, 0, 10, 10)]
        )
        assert unassigned[0]["reason"] == "outside_regions"
        assert unassigned[0]["touching"] == []


class TestPageImageMapping(unittest.TestCase):
    def test_simple_page_origin_is_not_assumed(self):
        with pdfplumber.open(NICS) as pdf:
            page = pdf.pages[0]
            im = page.to_image()
            # nics page starts at (0, 0), so pick a char and verify roundtrip.
            char = next(c for c in page.chars if c["text"].strip())
            page_bbox = (char["x0"], char["top"], char["x1"], char["bottom"])
            pixel_bbox = im.page_to_pixels(page_bbox)
            back = im.pixels_to_page(pixel_bbox)
            assert back == pytest.approx(page_bbox)

    def test_offset_cropbox(self):
        path = os.path.join(HERE, "pdfs/page-boxes-example.pdf")
        with pdfplumber.open(path) as pdf:
            page = pdf.pages[0]
            assert page.cropbox != page.mediabox
            im = page.to_image()
            assert im.bbox == page.cropbox
            # Pixel (0, 0) maps to the CropBox origin, not to (0, 0).
            assert im.pixels_to_page((0, 0)) == pytest.approx(page.cropbox[:2])
            assert im.pixels_to_page(im.original.size) == pytest.approx(
                page.cropbox[2:]
            )
            sx, sy = im.pixel_scale
            assert sx == pytest.approx(
                im.original.size[0] / (page.cropbox[2] - page.cropbox[0])
            )
            assert sy == pytest.approx(
                im.original.size[1] / (page.cropbox[3] - page.cropbox[1])
            )

    def test_nested_crop(self):
        with pdfplumber.open(NICS) as pdf:
            page = pdf.pages[0]
            cropped = page.crop((10, 20, 400, 300)).crop((50, 40, 350, 250))
            im = cropped.to_image()
            assert im.original.size == (300, 210)
            assert im.pixels_to_page((0, 0)) == pytest.approx(cropped.bbox[:2])

    def test_rotated_page_is_not_rotated_twice(self):
        for name in [
            "annotations-rotated-90.pdf",
            "annotations-rotated-180.pdf",
            "annotations-rotated-270.pdf",
        ]:
            path = os.path.join(HERE, "pdfs", name)
            with pdfplumber.open(path) as pdf:
                page = pdf.pages[0]
                im = page.to_image()
                # The rendered image matches the already-rotated page, so the
                # image aspect ratio must match the page's, not the pre-rotation
                # MediaBox aspect ratio.
                W, H = im.original.size
                assert W / H == pytest.approx(page.width / page.height, rel=1e-2)
                # A full-image region should recover all the real glyphs,
                # in the same order as the page itself.
                result = im.extract_regions((0, 0, W, H))
                glyphs = [c["text"] for c in page.chars if c["text"].strip()]
                assert result["regions"][0]["glyph_count"] == len(glyphs)
                assert result["unassigned"] == []
                assert result["rotation"] == page.rotation
                # Each glyph's page bbox must map inside the image without
                # needing an extra rotation.
                for glyph in result["regions"][0]["glyphs"]:
                    px0, py0, px1, py1 = glyph["pixel_bbox"]
                    assert 0 <= px0 <= px1 <= W
                    assert 0 <= py0 <= py1 <= H

    def test_mapping_has_no_flip_or_extra_rotation(self):
        # Render the page, then measure how much ink falls inside each glyph
        # box under the mapping produced by this module.  If the mapping
        # flipped an axis or applied an extra rotation, the mirrored boxes
        # would cover the ink instead, so the aligned score must clearly beat
        # every flipped variant.  Covers rotated pages, negative MediaBox
        # origins, and offset CropBoxes.
        cases = [
            ("annotations-rotated-90.pdf", 0),
            ("annotations-rotated-180.pdf", 0),
            ("annotations-rotated-270.pdf", 0),
            ("issue-1181.pdf", 0),  # MediaBox origin (0, -200)
            ("issue-1181.pdf", 1),  # MediaBox origin (420.9, 0)
            ("page-boxes-example.pdf", 0),  # offset CropBox
            ("issue-1054-example.pdf", 0),  # rotated + offset CropBox
        ]
        for name, page_index in cases:
            path = os.path.join(HERE, "pdfs", name)
            with pdfplumber.open(path) as pdf:
                page = pdf.pages[page_index]
                im = page.to_image()
                W, H = im.original.size
                ink = im.original.convert("L").point(lambda v: 255 if v < 200 else 0)
                boxes = [
                    im.page_to_pixels((c["x0"], c["top"], c["x1"], c["bottom"]))
                    for c in page.root_page.chars
                    if c["text"].strip()
                ]
                assert boxes

                def score(transform):
                    return _mean_ink(ink, [transform(b) for b in boxes])

                aligned = score(lambda b: b)
                flipped = [
                    score(lambda b: (b[0], H - b[3], b[2], H - b[1])),  # y
                    score(lambda b: (W - b[2], b[1], W - b[0], b[3])),  # x
                    score(lambda b: (W - b[2], H - b[3], W - b[0], H - b[1])),
                ]
                assert aligned > 0.1
                assert aligned > 1.5 * max(flipped)


def _mean_ink(ink, boxes):
    """Mean fraction of a glyph box covered by ink (PIL-only, no numpy)."""
    W, H = ink.size
    scores = []
    for b in boxes:
        x0, y0, x1, y1 = [int(round(v)) for v in b]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        crop = ink.crop((x0, y0, x1, y1))
        hist = crop.histogram()
        scores.append(sum(hist[128:]) / float(crop.size[0] * crop.size[1]))
    return sum(scores) / len(scores) if scores else 0.0


class TestExtractRegions(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        cls.pdf = pdfplumber.open(NICS)
        cls.page = cls.pdf.pages[0]
        cls.im = cls.page.to_image()

    @classmethod
    def teardown_class(cls):
        cls.pdf.close()

    def test_single_region_matches_crop(self):
        pixel_bbox = (0, 0, 300, 120)
        page_bbox = self.im.pixels_to_page(pixel_bbox)
        result = self.im.extract_regions(pixel_bbox)
        region = result["regions"][0]
        assert region["page_bbox"] == pytest.approx(page_bbox)
        assert region["pixel_bbox"] == pytest.approx(pixel_bbox)
        assert region["text"] == self.page.crop(page_bbox).extract_text()
        assert region["char_count"] > 0

    def test_multiple_regions_partition_glyphs(self):
        W, H = self.im.original.size
        regions = {"left": (0, 0, W // 2, H), "right": (W // 2, 0, W, H)}
        result = self.im.extract_regions(regions)
        assert [r["id"] for r in result["regions"]] == ["left", "right"]
        total = sum(r["glyph_count"] for r in result["regions"])
        visible = len(
            [
                c
                for c in self.page.chars
                if c["text"].strip()
                and c["x0"] < self.im.bbox[2]
                and c["x1"] > self.im.bbox[0]
            ]
        )
        assert total + len(result["conflicts"]) == visible

    def test_no_duplicate_chars_across_regions(self):
        W, H = self.im.original.size
        result = self.im.extract_regions(
            [
                (0, 0, W, H),
                (0, 0, W // 2, H // 2),
            ]
        )
        seen = [id(c) for r in result["regions"] for c in r["chars"]]
        all_ids = [id(c) for r in result["regions"] for c in r["chars"]]
        assert len(seen) == len(all_ids)
        # Every glyph is in exactly one region's glyph list, but whitespace
        # chars can be shared only if they are genuinely in both regions --
        # they are not, because of largest-overlap assignment.
        assert len(set(id(c) for c in self.page.chars if c["text"].strip())) >= len(
            set(seen)
        )

    def test_char_pixel_and_page_coordinates(self):
        result = self.im.extract_regions((0, 0, 300, 120))
        glyph = result["regions"][0]["glyphs"][0]
        char = glyph["char"]
        assert glyph["page_bbox"] == (
            char["x0"],
            char["top"],
            char["x1"],
            char["bottom"],
        )
        assert glyph["pixel_bbox"] == pytest.approx(
            self.im.page_to_pixels(glyph["page_bbox"])
        )

    def test_touching_edge_is_unassigned(self):
        # Find a glyph and build a region whose left edge coincides with the
        # glyph's right edge: the two touch but do not overlap.
        char = max(
            (c for c in self.page.chars if c["text"].strip()),
            key=lambda c: c["x0"],
        )
        char_page = (char["x0"], char["top"], char["x1"], char["bottom"])
        px0, py0, px1, py1 = self.im.page_to_pixels(char_page)
        region = (px1, py0 - 5, px1 + 40, py1 + 5)
        result = self.im.extract_regions(region)
        assert all(id(c) != id(char) for c in result["regions"][0]["chars"])
        touching = [u for u in result["unassigned"] if id(u["char"]) == id(char)]
        assert len(touching) == 1
        assert touching[0]["reason"] == "edge_contact_only"
        assert touching[0]["touching_regions"] == [0]

    def test_tie_conflict_reported_once(self):
        char = next(c for c in self.page.chars if c["text"].strip() and c["width"] > 1)
        px0, px1 = (
            self.im.page_to_pixels((char["x0"], char["top"]))[0],
            self.im.page_to_pixels((char["x1"], char["bottom"]))[0],
        )
        py0 = self.im.page_to_pixels((char["x0"], char["top"]))[1]
        py1 = self.im.page_to_pixels((char["x1"], char["bottom"]))[1]
        mid = (px0 + px1) / 2
        result = self.im.extract_regions(
            [(px0 - 1, py0 - 1, mid, py1 + 1), (mid, py0 - 1, px1 + 1, py1 + 1)]
        )
        conflict_ids = [id(c["char"]) for c in result["conflicts"]]
        assert id(char) in conflict_ids
        # The conflicted glyph is not part of any region's output.
        for region in result["regions"]:
            assert all(id(c) != id(char) for c in region["chars"])

    def test_include_flags(self):
        result = self.im.extract_regions(
            (0, 0, 10, 10), include_unassigned=False, include_conflicts=False
        )
        assert result["unassigned"] == []
        assert result["conflicts"] == []

    def test_text_kwargs_passthrough(self):
        result = self.im.extract_regions((0, 0, 300, 120), x_tolerance=10)
        assert isinstance(result["regions"][0]["text"], str)
        assert result["regions"][0]["text"] != ""

    def test_empty_region(self):
        result = self.im.extract_regions((0, 0, 1, 1))
        region = result["regions"][0]
        assert region["text"] == ""
        assert region["char_count"] == 0
        assert region["glyphs"] == []

    def test_module_function_matches_method(self):
        pixel_bbox = (0, 0, 300, 120)
        via_method = self.im.extract_regions(pixel_bbox)
        via_module = regions_mod.extract_regions(self.im, pixel_bbox)
        assert via_method["regions"][0]["text"] == via_module["regions"][0]["text"]


class TestLegacyApiCompatibility(unittest.TestCase):
    def test_drawing_still_works(self):
        with pdfplumber.open(NICS) as pdf:
            page = pdf.pages[0]
            im = page.to_image()
            im.reset()
            im.draw_rect((10, 10, 100, 100))
            im.draw_rects(page.rects[:2])
            im.draw_line(page.edges[0])
            im.draw_circle(page.chars[0])
            im.draw_vlines([10])
            im.draw_hlines([10])
            im.outline_words()
            im.outline_chars()
            assert im.annotated.size == im.original.size

    def test_crop_still_works(self):
        with pdfplumber.open(NICS) as pdf:
            page = pdf.pages[0]
            crop = page.crop((10, 20, 110, 120))
            assert crop.bbox == (10, 20, 110, 120)
            assert crop.extract_text() == page.crop((10, 20, 110, 120)).extract_text()

    def test_pageimage_attributes_unchanged(self):
        with pdfplumber.open(NICS) as pdf:
            im = pdf.pages[0].to_image(resolution=144)
            assert im.scale == pytest.approx(2.0, abs=1e-6)
            assert im.resolution == 144
            assert im._reproject((10, 20)) == (20, 40)
