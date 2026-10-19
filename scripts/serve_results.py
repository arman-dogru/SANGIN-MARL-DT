"""Serve an existing measured-results folder on loopback, without file-handle stat calls."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import mimetypes
from pathlib import Path
from urllib.parse import unquote, urlsplit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("output/results/learning-and-inference-20261006-01"))
    parser.add_argument("--port", type=int, default=8973)
    args = parser.parse_args()
    root = args.directory.resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            relative = unquote(urlsplit(self.path).path).lstrip("/") or "results.html"
            path = (root/relative).resolve()
            if not path.is_relative_to(root) or path.suffix not in (".html", ".png", ".pdf", ".csv", ".json", ".md"):
                self.send_error(403); return
            try:
                payload = path.read_bytes()
            except OSError:
                self.send_error(404); return
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length",str(len(payload)))
            self.send_header("Cache-Control","no-store")
            self.end_headers()
            self.wfile.write(payload)

    print(f"http://127.0.0.1:{args.port}/results.html",flush=True)
    ThreadingHTTPServer(("127.0.0.1",args.port),Handler).serve_forever()


if __name__ == "__main__":
    main()
