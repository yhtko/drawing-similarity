import json
import os
import sys

import open_clip
import torch
from PIL import Image, ImageOps


CENTER_CROP_BOX = (0.05, 0.10, 0.85, 0.80)
ROI_MAX_SIZE = int(os.environ.get("ROI_MAX_SIZE", "1200"))
ROI_THRESHOLD = int(os.environ.get("ROI_THRESHOLD", "235"))
ROI_MIN_COMPONENT_PIXELS = int(os.environ.get("ROI_MIN_COMPONENT_PIXELS", "20"))


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

    # Remove dominant frame/table ruling lines before connected-component scoring.
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
        "selected_components": len(selected),
    }


def choose_device() -> str:
    requested = os.environ.get("OPENCLIP_DEVICE", "auto").lower()
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("OPENCLIP_DEVICE=cuda was requested, but CUDA is not available")
        return "cuda"
    if requested != "auto":
        raise RuntimeError("OPENCLIP_DEVICE must be auto, cpu, or cuda")
    return "cuda" if torch.cuda.is_available() else "cpu"


def prepare_image(image_path: str):
    image_mode = os.environ.get("EMBED_IMAGE_MODE", "full").lower()
    image = Image.open(image_path).convert("RGB")
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
        image = image.crop((
            crop_box["left"],
            crop_box["top"],
            crop_box["right"],
            crop_box["bottom"],
        ))
    elif image_mode == "auto_roi":
        crop_box = find_auto_roi(image)
        if crop_box:
            image = image.crop((
                crop_box["left"],
                crop_box["top"],
                crop_box["right"],
                crop_box["bottom"],
            ))
    elif image_mode != "full":
        raise RuntimeError("EMBED_IMAGE_MODE must be full, center_crop, or auto_roi")

    return image, {
        "mode": image_mode,
        "source_width": source_width,
        "source_height": source_height,
        "width": image.size[0],
        "height": image.size[1],
        "crop_box": crop_box,
    }


def main() -> int:
    if len(sys.argv) != 2:
        print(json.dumps({"error": "usage: embed_openclip.py <image-path>"}), file=sys.stderr)
        return 2

    image_path = sys.argv[1]
    model_name = os.environ.get("OPENCLIP_MODEL", "ViT-B-32")
    pretrained = os.environ.get("OPENCLIP_PRETRAINED", "laion2b_s34b_b79k")
    device = choose_device()

    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.to(device)
    model.eval()

    image, image_info = prepare_image(image_path)
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        features = model.encode_image(image)
        features = features / features.norm(dim=-1, keepdim=True)

    vector = features.squeeze(0).detach().cpu().tolist()
    print(json.dumps({
        "provider": "openclip",
        "model": model_name,
        "pretrained": pretrained,
        "device": device,
        "dimension": len(vector),
        "image_mode": image_info["mode"],
        "image": image_info,
        "vector": vector,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
