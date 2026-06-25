import base64
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

import open_clip
import torch
import torch.nn.functional as functional

from embed_openclip import choose_device, prepare_image


HOST = os.environ.get("EMBEDDING_HOST", "127.0.0.1")
PORT = int(os.environ.get("EMBEDDING_PORT", "9090"))
PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "openclip").strip().lower()
OPENCLIP_MODEL = os.environ.get("OPENCLIP_MODEL", "ViT-B-32")
OPENCLIP_PRETRAINED = os.environ.get("OPENCLIP_PRETRAINED", "laion2b_s34b_b79k")
DINO_MODEL = os.environ.get("DINO_MODEL", "facebook/dinov2-small")

_STATE = None
_STATE_LOCK = threading.Lock()


def json_response(handler, status, payload):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class EmbeddingState:
    def __init__(self):
        torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
        self.provider = PROVIDER
        self.device = choose_device()
        if self.provider == "openclip":
            self.load_openclip()
        elif self.provider == "dinov2":
            self.load_dinov2()
        else:
            raise RuntimeError("EMBEDDING_PROVIDER must be openclip or dinov2 for embedding_server.py")

    def load_openclip(self):
        self.model_name = OPENCLIP_MODEL
        self.pretrained = OPENCLIP_PRETRAINED
        print(json.dumps({"event": "embedding_model_load_start", "provider": "openclip", "model": self.model_name, "pretrained": self.pretrained, "device": self.device}), flush=True)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained)
        self.model.to(self.device)
        self.model.eval()
        print(json.dumps({"event": "embedding_model_load_done", "provider": "openclip", "device": self.device}), flush=True)

    def load_dinov2(self):
        from transformers import AutoImageProcessor, AutoModel

        self.model_name = DINO_MODEL
        self.pretrained = ""
        print(json.dumps({"event": "embedding_model_load_start", "provider": "dinov2", "model": self.model_name, "device": self.device}), flush=True)
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(json.dumps({"event": "embedding_model_load_done", "provider": "dinov2", "device": self.device}), flush=True)

    def prepare_input_image(self, image_bytes, image_mode):
        previous_mode = os.environ.get("EMBED_IMAGE_MODE")
        os.environ["EMBED_IMAGE_MODE"] = image_mode
        try:
            return prepare_image(BytesIO(image_bytes))
        finally:
            if previous_mode is None:
                os.environ.pop("EMBED_IMAGE_MODE", None)
            else:
                os.environ["EMBED_IMAGE_MODE"] = previous_mode

    def embed_openclip(self, image):
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            features = self.model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).detach().cpu().tolist()

    def embed_dinov2(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]
            features = functional.normalize(features, p=2, dim=-1)
        return features.squeeze(0).detach().cpu().tolist()

    def embed(self, image_bytes, image_mode, rotation=0):
        image, image_info = self.prepare_input_image(image_bytes, image_mode)
        rotation = int(rotation or 0) % 360
        if rotation:
            image = image.rotate(-rotation, expand=True)
        image_info["rotation"] = rotation
        image_info["rotated_width"] = image.size[0]
        image_info["rotated_height"] = image.size[1]

        if self.provider == "openclip":
            vector = self.embed_openclip(image)
        else:
            vector = self.embed_dinov2(image)

        return {
            "provider": self.provider,
            "model": self.model_name,
            "pretrained": self.pretrained,
            "device": self.device,
            "dimension": len(vector),
            "image_mode": image_info["mode"],
            "image": image_info,
            "rotation": rotation,
            "vector": vector,
        }


def get_state():
    global _STATE
    if _STATE is not None:
        return _STATE
    with _STATE_LOCK:
        if _STATE is None:
            _STATE = EmbeddingState()
        return _STATE


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(json.dumps({"event": "embedding_http", "message": format % args}), flush=True)

    def do_GET(self):
        if self.path == "/health":
            state = _STATE
            json_response(self, 200, {
                "ok": True,
                "loaded": state is not None,
                "provider": PROVIDER,
                "model": state.model_name if state else (DINO_MODEL if PROVIDER == "dinov2" else OPENCLIP_MODEL),
                "pretrained": state.pretrained if state else ("" if PROVIDER == "dinov2" else OPENCLIP_PRETRAINED),
                "device": state.device if state else os.environ.get("OPENCLIP_DEVICE", "auto"),
            })
            return
        json_response(self, 404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if self.path != "/embed":
            json_response(self, 404, {"ok": False, "error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            image_bytes = base64.b64decode(payload.get("image_base64", ""), validate=True)
            image_mode = str(payload.get("image_mode") or os.environ.get("EMBED_IMAGE_MODE", "full")).lower()
            rotation = int(payload.get("rotation") or 0)
            result = get_state().embed(image_bytes, image_mode, rotation)
            json_response(self, 200, result)
        except Exception as exc:
            print(json.dumps({"event": "embedding_error", "provider": PROVIDER, "error": str(exc)}), flush=True)
            json_response(self, 500, {"ok": False, "error": str(exc), "provider": PROVIDER})


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(json.dumps({"event": "embedding_server_listening", "host": HOST, "port": PORT, "provider": PROVIDER}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
