import json
import os
import sys
from typing import List

from PIL import Image, ImageFilter


THRESHOLD = int(os.environ.get("SHAPE_THRESHOLD", "245"))
PROFILE_BINS = int(os.environ.get("SHAPE_PROFILE_BINS", "8"))


def fail(message: str) -> int:
    print(json.dumps({"error": message}), file=sys.stderr)
    return 2


def normalize_profile(values: List[int]) -> List[float]:
    total = sum(values)
    if total <= 0:
        return [0.0 for _ in values]
    return [round(value / total, 8) for value in values]


def parse_engine() -> str:
    engine = os.environ.get("SHAPE_ENGINE", "simple").strip().lower()
    if engine not in {"simple", "none"}:
        raise RuntimeError("SHAPE_ENGINE must be simple or none")
    return engine


def compute_profile(image_path: str) -> dict:
    image = Image.open(image_path).convert("L")
    width, height = image.size
    pixels = image.load()

    if width <= 0 or height <= 0:
        raise RuntimeError("empty image")

    left = width
    top = height
    right = -1
    bottom = -1
    ink = 0
    sum_x = 0
    sum_y = 0
    vertical = [0] * PROFILE_BINS
    horizontal = [0] * PROFILE_BINS
    total = width * height

    for y in range(height):
        for x in range(width):
            value = pixels[x, y]
            if value < THRESHOLD:
                ink += 1
                sum_x += x
                sum_y += y
                if x < left:
                    left = x
                if y < top:
                    top = y
                if x > right:
                    right = x
                if y > bottom:
                    bottom = y

                vx = min(PROFILE_BINS - 1, int(x * PROFILE_BINS / max(width, 1)))
                hy = min(PROFILE_BINS - 1, int(y * PROFILE_BINS / max(height, 1)))
                vertical[vx] += 1
                horizontal[hy] += 1

    if ink <= 0:
        bbox = None
        bbox_width = 0
        bbox_height = 0
        bbox_aspect_ratio = 0.0
        bbox_area_ratio = 0.0
        ink_ratio = 0.0
        centroid_x = 0.5
        centroid_y = 0.5
    else:
        bbox_width = max(1, right - left + 1)
        bbox_height = max(1, bottom - top + 1)
        bbox = {
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
            "width": bbox_width,
            "height": bbox_height,
        }
        bbox_aspect_ratio = round(bbox_width / bbox_height, 8)
        bbox_area_ratio = round((bbox_width * bbox_height) / total, 8)
        ink_ratio = round(ink / total, 8)
        centroid_x = round(sum_x / (ink * max(width, 1)), 8)
        centroid_y = round(sum_y / (ink * max(height, 1)), 8)

    edge_image = image.filter(ImageFilter.FIND_EDGES)
    edge_pixels = sum(1 for value in edge_image.getdata() if value < 80)
    edge_density = round(edge_pixels / total, 8)

    return {
        "engine": "simple",
        "mode": "simple",
        "sourceWidth": width,
        "sourceHeight": height,
        "bbox": bbox,
        "bboxAspectRatio": bbox_aspect_ratio,
        "bboxAreaRatio": bbox_area_ratio,
        "inkRatio": ink_ratio,
        "centroidX": centroid_x,
        "centroidY": centroid_y,
        "edgeDensity": edge_density,
        "verticalProfile": normalize_profile(vertical),
        "horizontalProfile": normalize_profile(horizontal),
    }


def main() -> int:
    if len(sys.argv) != 2:
        return fail("usage: extract_shape_profile.py <image-path>")

    image_path = sys.argv[1]

    try:
        engine = parse_engine()
        if engine == "none":
            print(json.dumps({
                "engine": "none",
                "mode": "none",
                "sourceWidth": 0,
                "sourceHeight": 0,
                "bbox": None,
                "bboxAspectRatio": 0.0,
                "bboxAreaRatio": 0.0,
                "inkRatio": 0.0,
                "centroidX": 0.5,
                "centroidY": 0.5,
                "edgeDensity": 0.0,
                "verticalProfile": [],
                "horizontalProfile": [],
            }))
            return 0

        print(json.dumps(compute_profile(image_path)))
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
