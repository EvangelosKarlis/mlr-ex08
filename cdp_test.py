"""End-to-end test of index.html in headless Edge via CDP.

Usage: py cdp_test.py <app_dir> <check_image.jpg> <expected.json> <model_info.json>
"""
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import websocket

app_dir, check_img, expected_path, info_path = sys.argv[1:5]
expected = json.load(open(expected_path))
classes = json.load(open(info_path))["classes"]
PORT = 8765
CDP_PORT = 9223


def pick_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


PORT = pick_port()
CDP_PORT = pick_port()
PROFILE = os.path.join(os.environ.get("TEMP", "."),
                       "edge-cdp-test-%d" % CDP_PORT)


class H(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


httpd = ThreadingHTTPServer(
    ("127.0.0.1", PORT),
    lambda *a, **kw: H(*a, directory=app_dir, **kw))
threading.Thread(target=httpd.serve_forever, daemon=True).start()

edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if not os.path.exists(edge):
    edge = r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
profile = PROFILE
proc = subprocess.Popen([
    edge, "--headless=new",
    "--remote-debugging-port=%d" % CDP_PORT,
    "--user-data-dir=" + profile,
    "--no-first-run", "--no-default-browser-check",
    "--use-fake-ui-for-media-stream",
    "--use-fake-device-for-media-stream",
    "about:blank",
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def http_json(port, path):
    import urllib.request
    with urllib.request.urlopen(
            "http://127.0.0.1:%d%s" % (port, path), timeout=5) as r:
        return json.load(r)


target_url = "http://127.0.0.1:%d/index.html" % PORT
deadline = time.time() + 60
targets = []
while time.time() < deadline:
    try:
        targets = http_json(CDP_PORT, "/json/list")
        if targets:
            break
    except Exception:
        time.sleep(0.5)
page = next(t for t in targets if t["type"] == "page")
ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=60)
mid = [0]


def send(method, **params):
    mid[0] += 1
    ws.send(json.dumps({"id": mid[0], "method": method, "params": params}))
    while True:
        msg = json.loads(ws.recv())
        if msg.get("id") == mid[0]:
            if "error" in msg:
                raise RuntimeError("%s: %s" % (method, msg["error"]))
            return msg.get("result", {})


def evaluate(expr, retry_for=0.0):
    end = time.time() + retry_for
    last = None
    while True:
        r = send("Runtime.evaluate", expression=expr,
                 returnByValue=True, awaitPromise=True)
        last = r.get("result", {}).get("value")
        if last not in (None, False, "") or time.time() > end:
            return last
        time.sleep(0.3)


send("Runtime.enable")
send("Page.enable")
send("Page.navigate", url=target_url)

ok = True

# 1. Model load
status = evaluate("document.getElementById('status').textContent",
                  retry_for=120)
print("status:", status)
ready = "ready" in (status or "").lower()
print("model ready:", ready)
ok &= ready

# 2. Photo upload of check_image.jpg through the real file input
send("DOM.enable")
doc = send("DOM.getDocument")
node = send("DOM.querySelector", nodeId=doc["root"]["nodeId"],
            selector="#fileInput")["nodeId"]
send("DOM.setFileInputFiles", files=[os.path.abspath(check_img)], nodeId=node)
got = evaluate("window.lastProbs ? JSON.stringify(window.lastProbs) : null",
               retry_for=60)
probs = json.loads(got)
errs = [abs(probs[i] - expected[c]) for i, c in enumerate(classes)]
worst = max(errs)
print("upload max abs err vs expected: %.3e" % worst)
print({c: round(probs[i], 5) for i, c in enumerate(classes)})
ok &= worst < 2e-3

# 3. Live loop: inject a synthetic camera stream (canvas.captureStream) and
#    verify the rAF loop keeps classifying frames.
loop_js = """
(async () => {
  const c = document.createElement('canvas');
  c.width = 400; c.height = 300;
  const g = c.getContext('2d');
  let n = 0;
  const draw = () => {
    g.fillStyle = n++ % 2 ? '#2e7d32' : '#8d6e63';
    g.fillRect(0, 0, 400, 300);
    g.fillStyle = '#fff';
    g.font = '40px sans-serif';
    g.fillText(String(n), 20, 60);
  };
  draw();
  setInterval(draw, 200);
  await window.__startCamera(c.captureStream(10));
  return document.getElementById('camBtn').textContent;
})()
"""
btn = send("Runtime.evaluate", expression=loop_js,
           returnByValue=True, awaitPromise=True)
btn = btn.get("result", {}).get("value")
t0 = evaluate("window.lastProbsAt || 0")
time.sleep(4)
t1 = evaluate("window.lastProbsAt || 0")
status2 = evaluate("document.getElementById('status').textContent")
print("live loop: btn=%r frames_classified_more=%s status=%r"
      % (btn, (t1 or 0) > (t0 or 0), status2))
ok &= (btn == "Stop camera") and (t1 or 0) > (t0 or 0)
if "error" in (status2 or "").lower():
    print("loop reported an error -> FAIL")
    ok = False

# 4. QR presence
qr = evaluate("document.getElementById('qrbox').querySelector('svg')!==null")
link = evaluate("document.getElementById('applink').href")
print("qr svg present:", qr, "link:", link)
ok &= bool(qr)

print("RESULT:", "PASS" if ok else "FAIL")

try:
    send("Browser.close")
except Exception:
    pass
try:
    ws.close()
except Exception:
    pass
try:
    proc.terminate()
except Exception:
    pass
httpd.shutdown()
sys.exit(0 if ok else 1)
