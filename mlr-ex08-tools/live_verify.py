"""Verify the LIVE Pages site: probabilities + QR decode. Closes the loop on
'check_image gives expected probs' and 'QR encodes the live URL'."""
import json
import os
import re
import socket
import subprocess
import sys
import time

import websocket
from PIL import Image
import cv2

LIVE = "https://evangeloskarlis.github.io/mlr-ex08/"
here = os.path.dirname(os.path.abspath(__file__))
check_img = sys.argv[1]
expected = json.load(open(sys.argv[2]))
classes = json.load(open(sys.argv[3]))["classes"]

with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    CDP_PORT = s.getsockname()[1]

edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if not os.path.exists(edge):
    edge = r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
proc = subprocess.Popen([
    edge, "--headless=new",
    "--remote-debugging-port=%d" % CDP_PORT,
    "--user-data-dir=" + os.path.join(os.environ.get("TEMP", "."),
                                      "edge-live-%d" % CDP_PORT),
    "--no-first-run", "--no-default-browser-check", "about:blank",
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import urllib.request


def http_json(path):
    with urllib.request.urlopen(
            "http://127.0.0.1:%d%s" % (CDP_PORT, path), timeout=5) as r:
        return json.load(r)


deadline = time.time() + 60
targets = []
while time.time() < deadline:
    try:
        targets = [t for t in http_json("/json/list") if t["type"] == "page"]
        if targets:
            break
    except Exception:
        time.sleep(0.5)
ws = websocket.create_connection(targets[0]["webSocketDebuggerUrl"], timeout=120)
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
    while True:
        r = send("Runtime.evaluate", expression=expr, returnByValue=True,
                 awaitPromise=True)
        v = r.get("result", {}).get("value")
        if v not in (None, False, "") or time.time() > end:
            return v
        time.sleep(0.3)


ok = True
send("Runtime.enable")
send("Page.enable")
send("Page.navigate", url=LIVE)

status = evaluate("document.getElementById('status').textContent",
                  retry_for=120)
print("live status:", status)
ok &= "ready" in (status or "").lower()

send("DOM.enable")
doc = send("DOM.getDocument")
node = send("DOM.querySelector", nodeId=doc["root"]["nodeId"],
            selector="#fileInput")["nodeId"]
send("DOM.setFileInputFiles", files=[os.path.abspath(check_img)], nodeId=node)
got = evaluate("window.lastProbs ? JSON.stringify(window.lastProbs) : null",
               retry_for=60)
probs = json.loads(got)
worst = max(abs(probs[i] - expected[c]) for i, c in enumerate(classes))
print("LIVE upload max abs err vs expected: %.3e" % worst)
print({c: round(probs[i], 6) for i, c in enumerate(classes)})
ok &= worst < 2e-3

# QR: reconstruct the rendered SVG as an image and decode it.
svg = evaluate("document.getElementById('qrbox').innerHTML")
link = evaluate("document.getElementById('applink').href")
print("applink href on live page:", link)
vb = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
# createSvgTag emits one <path> with 'M x,y l<cell>,0 ...' per dark module.
d = re.search(r'<path d="([^"]*)"', svg).group(1)
cell = float(re.search(r'l([\d.]+),0', d).group(1))
n = int(round(float(vb.group(1)) / cell))
mat = [[0] * n for _ in range(n)]
for x, y in re.findall(r'M([\d.]+),([\d.]+)', d):
    mat[int(round(float(y) / cell))][int(round(float(x) / cell))] = 1
scale = 10
img = Image.new("L", ((n + 8) * scale, (n + 8) * scale), 255)
for r in range(n):
    for c in range(n):
        if mat[r][c]:
            for dy in range(scale):
                for dx in range(scale):
                    img.putpixel(((c + 4) * scale + dx, (r + 4) * scale + dy), 0)
png = os.path.join(here, "qr_from_live.png")
img.save(png)
text, _, _ = cv2.QRCodeDetector().detectAndDecode(cv2.imread(png))
print("decoded QR content:", repr(text))
print("matches live URL:", text == LIVE)
ok &= (text == LIVE) and (link.rstrip("/") + "/" == LIVE)

# model fetchable from Pages
code = 0
try:
    with urllib.request.urlopen(LIVE + "leaf_model.onnx", timeout=30) as r:
        code = r.status
except Exception as e:
    print("model fetch error:", e)
print("leaf_model.onnx served with HTTP", code)
ok &= code == 200

print("LIVE RESULT:", "PASS" if ok else "FAIL")
try:
    send("Browser.close")
except Exception:
    pass
try:
    proc.terminate()
except Exception:
    pass
sys.exit(0 if ok else 1)
