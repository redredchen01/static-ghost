"""Interactive watermark region picker via local HTML page.

Opens a browser with a video frame where the user can draw a rectangle
to select the watermark region. Returns coordinates as Region.
"""
from __future__ import annotations

import base64
import http.server
import json
import os
import threading
import webbrowser

import cv2

from static_ghost.detector import Region

_HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Static Ghost — Select Watermark Region</title>
<style>
  body { margin:0; background:#111; color:#fff; font-family:system-ui; display:flex; flex-direction:column; align-items:center; }
  h2 { margin:12px 0 8px; }
  p { margin:4px 0; color:#aaa; font-size:14px; }
  canvas { cursor:crosshair; border:1px solid #444; max-width:95vw; }
  #coords { font-family:monospace; font-size:18px; color:#0f0; margin:8px 0; min-height:24px; }
  button { padding:10px 32px; font-size:16px; background:#0a0; color:#fff; border:none; border-radius:6px; cursor:pointer; margin:8px; }
  button:hover { background:#0c0; }
  button:disabled { background:#444; cursor:not-allowed; }
  .done-msg { margin-top:40vh; text-align:center; }
</style></head><body>
<h2>Draw a rectangle around the watermark</h2>
<p>Click and drag to select. You can redraw as many times as needed.</p>
<div id="coords">No selection yet</div>
<canvas id="c"></canvas>
<button id="btn" disabled onclick="submit()">Confirm Selection</button>
<script>
const img = new Image();
img.onload = function() {
  const c = document.getElementById('c');
  const scale = Math.min(1, window.innerWidth * 0.95 / img.width);
  c.width = img.width * scale; c.height = img.height * scale;
  const ctx = c.getContext('2d');
  ctx.drawImage(img, 0, 0, c.width, c.height);
  let sx,sy,drawing=false,rect=null;
  c.onmousedown = function(e) { sx=e.offsetX; sy=e.offsetY; drawing=true; };
  c.onmousemove = function(e) {
    if(!drawing) return;
    ctx.drawImage(img, 0, 0, c.width, c.height);
    const w=e.offsetX-sx, h=e.offsetY-sy;
    ctx.strokeStyle='#f00'; ctx.lineWidth=2; ctx.strokeRect(sx,sy,w,h);
  };
  c.onmouseup = function(e) {
    drawing=false;
    const x=Math.round(Math.min(sx,e.offsetX)/scale), y=Math.round(Math.min(sy,e.offsetY)/scale);
    const w=Math.round(Math.abs(e.offsetX-sx)/scale), h=Math.round(Math.abs(e.offsetY-sy)/scale);
    if(w>5 && h>5) {
      rect={x:x,y:y,w:w,h:h};
      document.getElementById('coords').textContent='x='+x+', y='+y+', w='+w+', h='+h;
      document.getElementById('btn').disabled=false;
    }
  };
};
img.src = "FRAME_DATA_URL";
function submit() {
  fetch('/submit', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(rect)})
    .then(function() {
      var msg = document.createElement('h2');
      msg.className = 'done-msg';
      msg.textContent = 'Region saved. You can close this tab.';
      document.body.replaceChildren(msg);
    });
}
</script></body></html>"""


def pick_region(frame_path: str, port: int = 18923) -> Region | None:
    """Open browser for user to draw watermark region. Returns Region or None."""
    with open(frame_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(frame_path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    data_url = f"data:{mime};base64,{b64}"

    html = _HTML_TEMPLATE.replace("FRAME_DATA_URL", data_url)
    result = {"region": None}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html.encode())

        def do_POST(self):
            length = int(self.headers["Content-Length"])
            data = json.loads(self.rfile.read(length))
            result["region"] = data
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
            threading.Thread(target=self.server.shutdown, daemon=True).start()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"Opening region picker at {url}")
    webbrowser.open(url)

    server.serve_forever()
    server.server_close()

    r = result["region"]
    if r:
        return Region(x=r["x"], y=r["y"], w=r["w"], h=r["h"], confidence=1.0)
    return None
