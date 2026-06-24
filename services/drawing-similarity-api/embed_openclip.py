import json
import os
import sys

import open_clip
import torch
from PIL import Image


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

    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

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
        "vector": vector,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
