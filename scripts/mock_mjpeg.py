"""A local MJPEG source for checking the ESP32 adapter without hardware."""
import argparse
import io
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image, ImageDraw


class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/stream":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        index = 0
        try:
            while True:
                image = Image.new("RGB", (640, 480), (35, 45, 65))
                ImageDraw.Draw(image).text((30 + index % 450, 180),
                                          f"MJPEG TEST FRAME {index} - not a person", fill="white")
                buffer = io.BytesIO()
                image.save(buffer, "JPEG", quality=80)
                payload = buffer.getvalue()
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(payload)).encode() + b"\r\n\r\n" + payload + b"\r\n")
                self.wfile.flush()
                index += 1
                time.sleep(0.1)
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    print(f"MJPEG test source: http://127.0.0.1:{args.port}/stream", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), StreamHandler).serve_forever()
