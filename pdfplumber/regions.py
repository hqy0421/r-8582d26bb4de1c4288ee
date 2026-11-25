"""Extract the glyphs located inside pixel-space regions of a ``PageImage``.

A region is a rectangle drawn on the rendered page image, expressed in image
pixel coordinates (origin at the image's top-left corner).  The functions in
this module translate those rectangles back into the page coordinate system
used by ``pdfplumber`` and then assign the page's actual ``char`` objects to
the regions that contain them.

Two details are worth calling out:

* The page coordinate system and the rendered image share the same
  orientation.  Page rotation has already been applied by both pdfminer and
  PDFium, so no rotation is applied here.  Doing so would rotate rotated
  pages twice.
* Coordinates are never assumed to start at ``(0, 0)``.  The image's own
  displayed page bounds (``PageImage.bbox``) and its own pixel dimensions are
  used for the translation, which keeps offset CropBoxes and nested crops
  correct.

A glyph is assigned to a region when the two rectangles share a strictly
positive area.  Merely touching along an edge (or at a corner) does not count
as an intersection.  When a glyph overlaps more than one region, it is
assigned to the region with the largest intersection area.  If several
regions share that largest area, the glyph is not assigned to any of them;
it is reported once as a conflict instead of being duplicated in the output.
"""

import numbers
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from . import utils
from ._typing import T_bbox, T_num, T_obj, T_obj_list, T_point

if TYPE_CHECKING:  # pragma: nocover
    from .display import PageImage

# A rectangle whose width or height is at most this many page units is
# treated as touching rather than overlapping.  This absorbs the tiny
# floating-point residue introduced by the pixel <-> page conversion.
EDGE_TOLERANCE = 1e-9

# Relative tolerance used to decide whether two overlap areas are ties.
# Exact ties arise, for example, when a glyph is centered on the border
# between two equally-sized regions; binary floating point can still make the
# two computed areas differ by an ulp.
TIE_TOLERANCE = 1e-9

T_coord = Union[T_point, T_bbox]
T_regions = Union[
    T_bbox,
    T_obj,
    Sequence[Union[T_bbox, T_obj]],
    Mapping[Any, Union[T_bbox, T_obj]],
]

_BBOX_KEYS = ("x0", "top", "x1", "bottom")
_ALT_BBOX_KEYS = (
    ("x0", "y0", "x1", "y1"),
    ("left", "top", "right", "bottom"),
    ("left", "bottom", "right", "top"),
)


def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _bbox_key_order(value: Mapping[Any, Any]) -> Optional[Tuple[str, str, str, str]]:
    if all(k in value for k in _BBOX_KEYS):
        return _BBOX_KEYS
    for keys in _ALT_BBOX_KEYS:
        if all(k in value for k in keys):
            return keys
    return None


def _is_bbox_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and _bbox_key_order(value) is not None


def coerce_bbox(value: Any, label: str = "region") -> T_bbox:
    """Normalize a bbox-like value into an ``(x0, top, x1, bottom)`` tuple.

    Accepts a 4-item sequence/iterable or a mapping using
    ``x0/top/x1/bottom``, ``x0/y0/x1/y1``, or ``left/top/right/bottom`` keys.
    Reversed coordinates are sorted, and zero-area rectangles raise
    ``ValueError``.
    """
    if isinstance(value, Mapping):
        keys = _bbox_key_order(value)
        if keys is None:
            raise ValueError(
                f"{label} mapping must contain x0/top/x1/bottom keys, not "
                f"{sorted(value.keys())}"
            )
        raw = tuple(value[k] for k in keys)
    else:
        if isinstance(value, (str, bytes)):
            raise TypeError(
                f"{label} must be a 4-item sequence or a mapping, not "
                f"{type(value).__name__}"
            )
        try:
            raw = tuple(value)
        except TypeError:
            raise TypeError(
                f"{label} must be a 4-item sequence or a mapping with "
                f"x0/top/x1/bottom keys, not {type(value).__name__}"
            )
        if len(raw) != 4:
            raise ValueError(f"{label} must contain exactly 4 coordinates")

    if not all(_is_number(v) for v in raw):  # pragma: nocover
        raise TypeError(f"{label} contains non-numeric coordinate(s): {raw}")

    x0, x1 = sorted((float(raw[0]), float(raw[2])))
    top, bottom = sorted((float(raw[1]), float(raw[3])))
    if (x1 - x0) <= 0 or (bottom - top) <= 0:
        raise ValueError(f"{label} {tuple(raw)} has zero area")
    return (x0, top, x1, bottom)


def normalize_regions(regions: T_regions) -> List[Dict[str, Any]]:
    """Return a list of ``{"id": ..., "pixel_bbox": ...}`` region records.

    Regions may be provided as:

    * a single bbox (a 4-item sequence or a bbox mapping);
    * a sequence of bboxes and/or bbox mappings;
    * a mapping of region ids to bboxes (or to bbox mappings);
    * a pandas ``DataFrame`` with ``x0/top/x1/bottom`` columns.

    Region ids come from the mapping keys, from an ``"id"``/``"name"`` key in
    a region mapping, or from the item's position in the sequence.
    """
    if isinstance(regions, Mapping):
        if _is_bbox_mapping(regions):
            items: List[Tuple[Any, Any]] = [(0, regions)]
        else:
            items = list(regions.items())
    elif (
        isinstance(regions, Sequence)
        and not isinstance(regions, (str, bytes))
        and len(regions) == 4
        and all(_is_number(v) for v in regions)
    ):
        items = [(0, regions)]
    else:
        items = list(enumerate(utils.to_list(regions)))

    normalized: List[Dict[str, Any]] = []
    for key, value in items:
        pixel_bbox = coerce_bbox(value, label=f"region {key!r}")
        region_id: Any = key
        if isinstance(value, Mapping):
            region_id = value.get("id", value.get("name", key))
        normalized.append({"id": region_id, "pixel_bbox": pixel_bbox})
    return normalized


def glyph_bbox(char: T_obj) -> T_bbox:
    """Return a char's bbox as an ``(x0, top, x1, bottom)`` tuple."""
    return (char["x0"], char["top"], char["x1"], char["bottom"])


def _is_glyph(char: T_obj) -> bool:
    """Whether a char is a real glyph, as opposed to layout-implied space."""
    text = char.get("text")
    return isinstance(text, str) and bool(text.strip())


def overlap_kind(
    a: T_bbox, b: T_bbox, tolerance: float = EDGE_TOLERANCE
) -> Tuple[str, float]:
    """Classify the relationship between two bboxes.

    Returns ``(kind, area)`` where ``kind`` is one of:

    * ``"overlap"`` -- the rectangles share a strictly positive area;
    * ``"touch"`` -- they touch along an edge or at a corner only;
    * ``"none"`` -- they are separated.

    ``area`` is the intersection area for ``"overlap"`` and ``0.0`` otherwise.
    """
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width > tolerance and height > tolerance:
        return ("overlap", width * height)
    if width >= -tolerance and height >= -tolerance:
        return ("touch", 0.0)
    return ("none", 0.0)


def _areas_tied(a: T_num, b: T_num, tolerance: float) -> bool:
    return abs(a - b) <= tolerance * max(1.0, abs(a), abs(b))


def assign_glyphs_to_regions(
    glyphs: T_obj_list,
    region_bboxes: Sequence[T_bbox],
    tolerance: float = EDGE_TOLERANCE,
    tie_tolerance: float = TIE_TOLERANCE,
) -> Tuple[
    List[List[Tuple[T_obj, float]]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """Assign glyphs to page-space regions.

    Returns ``(assignments, unassigned, conflicts)``, where ``assignments`` is
    a list (parallel to ``region_bboxes``) of ``(glyph, overlap_area)``
    tuples.  Unassigned glyphs carry a ``reason`` of ``"outside_regions"``,
    ``"edge_contact_only"``, or ``"zero_area_glyph"``.  Tied glyphs are
    returned in ``conflicts`` with reason ``"overlap_tie"`` and are *not*
    added to any region, so callers never see the same glyph twice.
    """
    assignments: List[List[Tuple[T_obj, float]]] = [[] for _ in region_bboxes]
    unassigned: List[Dict[str, Any]] = []
    conflicts: List[Dict[str, Any]] = []

    for glyph in glyphs:
        bbox = glyph_bbox(glyph)
        if (bbox[2] - bbox[0]) <= tolerance or (bbox[3] - bbox[1]) <= tolerance:
            unassigned.append(
                {"char": glyph, "reason": "zero_area_glyph", "touching": []}
            )
            continue

        overlaps: List[Tuple[int, float]] = []
        touching: List[int] = []
        for i, region_bbox in enumerate(region_bboxes):
            kind, area = overlap_kind(bbox, region_bbox, tolerance=tolerance)
            if kind == "overlap":
                overlaps.append((i, area))
            elif kind == "touch":
                touching.append(i)

        if not overlaps:
            unassigned.append(
                {
                    "char": glyph,
                    "reason": "edge_contact_only" if touching else "outside_regions",
                    "touching": touching,
                }
            )
            continue

        largest = max(area for _, area in overlaps)
        best = [
            (i, area)
            for i, area in overlaps
            if _areas_tied(area, largest, tie_tolerance)
        ]
        if len(best) > 1:
            conflicts.append(
                {
                    "char": glyph,
                    "reason": "overlap_tie",
                    "overlap_areas": overlaps,
                }
            )
        else:
            assignments[best[0][0]].append((glyph, best[0][1]))

    return (assignments, unassigned, conflicts)


def _visible_chars(chars: T_obj_list, display_bbox: T_bbox) -> T_obj_list:
    """Filter chars down to those at least partly within the image.

    Chars entirely outside the image's displayed page bounds are not part of
    "this image".  Whitespace chars are kept: they are real char objects, and
    they carry the spacing that keeps each region's text faithful to
    ``Page.extract_text``.
    """
    visible: T_obj_list = []
    for char in chars:
        kind, _ = overlap_kind(glyph_bbox(char), display_bbox)
        if kind != "none":
            visible.append(char)
    return visible


def _diagnostic(
    record: Dict[str, Any],
    regions: List[Dict[str, Any]],
    page_image: "PageImage",
) -> Dict[str, Any]:
    char = record["char"]
    bbox = glyph_bbox(char)
    output: Dict[str, Any] = {
        "char": char,
        "text": char.get("text"),
        "page_bbox": bbox,
        "pixel_bbox": page_image.page_to_pixels(bbox),
        "reason": record["reason"],
    }
    if record.get("touching"):
        output["touching_regions"] = [regions[i]["id"] for i in record["touching"]]
    if "overlap_areas" in record:
        output["candidates"] = [
            {
                "id": regions[i]["id"],
                "index": i,
                "overlap_area": area,
            }
            for i, area in record["overlap_areas"]
        ]
    return output


def _region_text(chars: T_obj_list, page_bbox: T_bbox, **kwargs: Any) -> str:
    """Build a region's text the same way ``Page.extract_text`` does."""
    defaults: Dict[str, Any] = {"layout_bbox": page_bbox}
    if "layout_width" not in kwargs and "layout_width_chars" not in kwargs:
        defaults["layout_width"] = page_bbox[2] - page_bbox[0]
    if "layout_height" not in kwargs and "layout_height_chars" not in kwargs:
        defaults["layout_height"] = page_bbox[3] - page_bbox[1]
    return utils.chars_to_textmap(chars, **{**defaults, **kwargs}).as_string


def extract_regions(
    page_image: "PageImage",
    regions: T_regions,
    *,
    include_unassigned: bool = True,
    include_conflicts: bool = True,
    tie_tolerance: float = TIE_TOLERANCE,
    **text_kwargs: Any,
) -> Dict[str, Any]:
    """Extract glyphs from pixel-space regions of ``page_image``.

    See ``PageImage.extract_regions`` for the full description.  This
    function is the implementation behind that method and can also be called
    directly with ``pdfplumber.regions.extract_regions(im, regions)``.
    """
    normalized = normalize_regions(regions)

    page_bboxes = [
        page_image.pixels_to_page(region["pixel_bbox"]) for region in normalized
    ]
    # Use the root page's chars so that glyph boxes are the true, unclipped
    # boxes even when the image was built from a (possibly nested) crop.
    chars = _visible_chars(page_image.page.root_page.chars, page_image.bbox)

    assignments, unassigned, conflicts = assign_glyphs_to_regions(
        chars,
        page_bboxes,
        tie_tolerance=tie_tolerance,
    )

    region_results: List[Dict[str, Any]] = []
    for region, page_bbox, assigned in zip(normalized, page_bboxes, assignments):
        region_chars = [char for char, _ in assigned]
        text = (
            _region_text(region_chars, page_bbox, **text_kwargs) if region_chars else ""
        )
        glyph_records = [
            {
                "char": char,
                "text": char.get("text"),
                "page_bbox": glyph_bbox(char),
                "pixel_bbox": page_image.page_to_pixels(glyph_bbox(char)),
                "overlap_area": area,
            }
            for char, area in assigned
            if _is_glyph(char)
        ]
        region_results.append(
            {
                "id": region["id"],
                "pixel_bbox": region["pixel_bbox"],
                "page_bbox": page_bbox,
                "text": text,
                "chars": region_chars,
                "char_count": len(region_chars),
                "glyphs": glyph_records,
                "glyph_count": len(glyph_records),
            }
        )

    scale_x, scale_y = page_image.pixel_scale
    return {
        "page_number": page_image.page.page_number,
        "rotation": page_image.page.rotation,
        "page_bbox": tuple(page_image.bbox),
        "image_size": tuple(page_image.original.size),
        "scale": (scale_x, scale_y),
        "regions": region_results,
        "unassigned": (
            [
                _diagnostic(r, normalized, page_image)
                for r in unassigned
                if _is_glyph(r["char"])
            ]
            if include_unassigned
            else []
        ),
        "conflicts": (
            [
                _diagnostic(r, normalized, page_image)
                for r in conflicts
                if _is_glyph(r["char"])
            ]
            if include_conflicts
            else []
        ),
    }


def _convert_coord(
    value: Union[T_coord, T_obj],
    mapper: Any,
) -> Union[T_point, T_bbox]:
    """Apply ``mapper`` to a point, bbox, or bbox mapping."""
    mapped: Union[T_point, T_bbox]
    if isinstance(value, Mapping):
        mapped = mapper(coerce_bbox(value, label="coordinate"))
        return mapped
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(
            "Coordinate must be a 2-item point, 4-item bbox, or bbox mapping"
        )
    coords = tuple(value)
    if len(coords) == 2:
        if not all(_is_number(v) for v in coords):  # pragma: nocover
            raise TypeError(f"Point contains non-numeric coordinate(s): {coords}")
        mapped = mapper((float(coords[0]), float(coords[1])))
    elif len(coords) == 4:
        mapped = mapper(coerce_bbox(coords, label="coordinate"))
    else:
        raise ValueError("Coordinate must contain either 2 or 4 values")
    return mapped


def pixel_scale(page_image: "PageImage") -> Tuple[float, float]:
    """Return ``(pixels_per_page_unit_x, pixels_per_page_unit_y)``.

    The values are computed from the image's actual pixel size and from the
    page-space bounds of the area the image displays.
    """
    width, height = page_image.original.size
    x0, top, x1, bottom = page_image.bbox
    if (x1 - x0) <= 0 or (bottom - top) <= 0:  # pragma: nocover
        raise ValueError(f"Cannot map image to zero-area page bounds {page_image.bbox}")
    return (width / (x1 - x0), height / (bottom - top))


def pixels_to_page(page_image: "PageImage", value: Union[T_coord, T_obj]) -> Any:
    """Map pixel-space coordinate(s) to page space, using the image bounds."""
    scale_x, scale_y = pixel_scale(page_image)
    bbox_x0, bbox_top = page_image.bbox[0], page_image.bbox[1]

    def mapper(coords: Sequence[float]) -> Union[T_point, T_bbox]:
        if len(coords) == 2:
            return (
                bbox_x0 + coords[0] / scale_x,
                bbox_top + coords[1] / scale_y,
            )
        return (
            bbox_x0 + coords[0] / scale_x,
            bbox_top + coords[1] / scale_y,
            bbox_x0 + coords[2] / scale_x,
            bbox_top + coords[3] / scale_y,
        )

    return _convert_coord(value, mapper)


def page_to_pixels(page_image: "PageImage", value: Union[T_coord, T_obj]) -> Any:
    """Map page-space coordinate(s) to pixels, using the image bounds."""
    scale_x, scale_y = pixel_scale(page_image)
    bbox_x0, bbox_top = page_image.bbox[0], page_image.bbox[1]

    def mapper(coords: Sequence[float]) -> Union[T_point, T_bbox]:
        if len(coords) == 2:
            return (
                (coords[0] - bbox_x0) * scale_x,
                (coords[1] - bbox_top) * scale_y,
            )
        return (
            (coords[0] - bbox_x0) * scale_x,
            (coords[1] - bbox_top) * scale_y,
            (coords[2] - bbox_x0) * scale_x,
            (coords[3] - bbox_top) * scale_y,
        )

    return _convert_coord(value, mapper)
