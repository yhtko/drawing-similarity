import json
import os
import sys
from typing import List

from PIL import Image, ImageFilter, ImageOps


THRESHOLD = int(os.environ.get("SHAPE_THRESHOLD", "245"))
PROFILE_BINS = int(os.environ.get("SHAPE_PROFILE_BINS", "8"))
CENTER_CROP_BOX = (0.05, 0.10, 0.85, 0.80)
ROI_MAX_SIZE = int(os.environ.get("ROI_MAX_SIZE", "1200"))
ROI_THRESHOLD = int(os.environ.get("ROI_THRESHOLD", "235"))
ROI_MIN_COMPONENT_PIXELS = int(os.environ.get("ROI_MIN_COMPONENT_PIXELS", "20"))


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


def find_auto_roi(source_image: Image.Image):
    gray = ImageOps.grayscale(source_image)
    source_width, source_height = gray.size
    scale = min(1.0, ROI_MAX_SIZE / max(source_width, source_height))
    work_width = max(1, int(source_width * scale))
    work_height = max(1, int(source_height * scale))
    work = gray.resize((work_width, work_height), Image.Resampling.BILINEAR)
    pixels = work.load()

    dark = set()
    row_counts = [0] * work_height
    col_counts = [0] * work_width
    for y in range(work_height):
        for x in range(work_width):
            if pixels[x, y] < ROI_THRESHOLD:
                dark.add((x, y))
                row_counts[y] += 1
                col_counts[x] += 1

    if not dark:
        return None

    noisy_rows = {index for index, count in enumerate(row_counts) if count / work_width > 0.45}
    noisy_cols = {index for index, count in enumerate(col_counts) if count / work_height > 0.45}
    candidates = {(x, y) for (x, y) in dark if y not in noisy_rows and x not in noisy_cols}
    if not candidates:
        candidates = dark

    visited = set()
    components = []
    for point in list(candidates):
        if point in visited:
            continue
        stack = [point]
        visited.add(point)
        left = right = point[0]
        top = bottom = point[1]
        count = 0
        while stack:
            x, y = stack.pop()
            count += 1
            left = min(left, x)
            right = max(right, x)
            top = min(top, y)
            bottom = max(bottom, y)
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                neighbor = (nx, ny)
                if neighbor in candidates and neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        if count >= ROI_MIN_COMPONENT_PIXELS:
            components.append((left, top, right, bottom, count))

    if not components:
        return None

    page_area = work_width * work_height
    scored = []
    for left, top, right, bottom, count in components:
        width = right - left + 1
        height = bottom - top + 1
        area_ratio = (width * height) / page_area
        if area_ratio > 0.75:
            continue
        center_x = (left + right) / (2 * work_width)
        center_y = (top + bottom) / (2 * work_height)
        centrality = 1 - min(((center_x - 0.48) ** 2 + (center_y - 0.43) ** 2) ** 0.5, 1)
        bottom_right_penalty = 0.55 if center_x > 0.62 and center_y > 0.62 else 1.0
        score = count * (0.65 + 0.35 * centrality) * bottom_right_penalty
        scored.append((score, left, top, right, bottom, count))

    if not scored:
        return None

    scored.sort(reverse=True)
    selected = scored[: min(8, len(scored))]
    left = min(item[1] for item in selected)
    top = min(item[2] for item in selected)
    right = max(item[3] for item in selected)
    bottom = max(item[4] for item in selected)

    pad_x = max(8, int((right - left + 1) * 0.08))
    pad_y = max(8, int((bottom - top + 1) * 0.08))
    left = max(0, left - pad_x)
    top = max(0, top - pad_y)
    right = min(work_width - 1, right + pad_x)
    bottom = min(work_height - 1, bottom + pad_y)

    inv_scale = 1 / scale
    return {
        "left": max(0, int(left * inv_scale)),
        "top": max(0, int(top * inv_scale)),
        "right": min(source_width, int((right + 1) * inv_scale)),
        "bottom": min(source_height, int((bottom + 1) * inv_scale)),
        "components": len(components),
        "selectedComponents": len(selected),
    }


def prepare_image(image_path: str):
    image_mode = os.environ.get("SHAPE_IMAGE_MODE", os.environ.get("EMBED_IMAGE_MODE", "full")).strip().lower()
    image = Image.open(image_path).convert("L")
    source_width, source_height = image.size
    crop_box = None

    if image_mode == "center_crop":
        left_ratio, top_ratio, right_ratio, bottom_ratio = CENTER_CROP_BOX
        crop_box = {
            "left": int(source_width * left_ratio),
            "top": int(source_height * top_ratio),
            "right": int(source_width * right_ratio),
            "bottom": int(source_height * bottom_ratio),
        }
        image = image.crop((crop_box["left"], crop_box["top"], crop_box["right"], crop_box["bottom"]))
    elif image_mode == "auto_roi":
        crop_box = find_auto_roi(image)
        if crop_box:
            image = image.crop((crop_box["left"], crop_box["top"], crop_box["right"], crop_box["bottom"]))
    elif image_mode != "full":
        raise RuntimeError("SHAPE_IMAGE_MODE must be full, center_crop, or auto_roi")

    return image, {
        "mode": image_mode,
        "sourceWidth": source_width,
        "sourceHeight": source_height,
        "cropBox": crop_box,
    }


def compute_profile(image_path: str) -> dict:
    image, image_info = prepare_image(image_path)
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
        "mode": image_info["mode"],
        "sourceWidth": image_info["sourceWidth"],
        "sourceHeight": image_info["sourceHeight"],
        "width": width,
        "height": height,
        "cropBox": image_info["cropBox"],
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
