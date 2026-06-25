import base64
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

import open_clip
import torch

from embed_openclip import choose_device, prepare_image


HOST = os.environ.get("EMBEDDING_HOST", "127.0.0.1")
PORT = int(os.environ.get("EMBEDDING_PORT", "9090"))
MODEL_NAME = os.environ.get("OPENCLIP_MODEL", "ViT-B-32")
PRETRAINED = os.environ.get("OPENCLIP_PRETRAINED", "laion2b_s34b_b79k")


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
        self.device = choose_device()
        print(json.dumps({
            "event": "embedding_model_load_start",
            "provider": "openclip",
            "model": MODEL_NAME,
            "pretrained": PRETRAINED,
            "device": self.device,
        }), flush=True)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
        self.model.to(self.device)
        self.model.eval()
        print(json.dumps({"event": "embedding_model_load_done", "device": self.device}), flush=True)

    def embed(self, image_bytes, image_mode):
        previous_mode = os.environ.get("EMBED_IMAGE_MODE")
        os.environ["EMBED_IMAGE_MODE"] = image_mode
        try:
            image, image_info = prepare_image(BytesIO(image_bytes))
        finally:
            if previous_mode is None:
                os.environ.pop("EMBED_IMAGE_MODE", None)
            else:
                os.environ["EMBED_IMAGE_MODE"] = previous_mode

        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            features = self.model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)

        vector = features.squeeze(0).detach().cpu().tolist()
        return {
            "provider": "openclip",
            "model": MODEL_NAME,
            "pretrained": PRETRAINED,
            "device": self.device,
            "dimension": len(vector),
            "image_mode": image_info["mode"],
            "image": image_info,
            "vector": vector,
        }


STATE = EmbeddingState()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(json.dumps({"event": "embedding_http", "message": format % args}), flush=True)

    def do_GET(self):
        if self.path == "/health":
            json_response(self, 200, {
                "ok": True,
                "provider": "openclip",
                "model": MODEL_NAME,
                "pretrained": PRETRAINED,
                "device": STATE.device,
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
            result = STATE.embed(image_bytes, image_mode)
            json_response(self, 200, result)
        except Exception as exc:
            print(json.dumps({"event": "embedding_error", "error": str(exc)}), flush=True)
            json_response(self, 500, {"ok": False, "error": str(exc)})


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(json.dumps({"event": "embedding_server_listening", "host": HOST, "port": PORT}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
