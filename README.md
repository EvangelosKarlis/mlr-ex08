# mlr-ex08 — Live Leaf Classifier (web)

**Open the app:** <https://evangeloskarlis.github.io/mlr-ex08/>

Classifies leaves in real time from your phone's rear camera (or an uploaded
photo) using a small ONNX CNN, entirely in the browser — no server, no install.

- One `index.html`, plain JavaScript, [onnxruntime-web] 1.31 (WASM) from a CDN.
- Works in Safari on iPhone (needs HTTPS — provided by GitHub Pages — and the
  “Start camera” button for the camera permission).
- Classes: `apple`, `cherry`, `chestnut`, `maple`.

## Preprocessing (exactly as described in `model_info.json`)

1. Central square crop of the camera frame.
2. Resize to 96 × 96 (`input_size`).
3. RGB scaled to 0–1.
4. Per-channel standardization with `mean = [0.485, 0.456, 0.406]`,
   `std = [0.229, 0.224, 0.225]`.
5. Input tensor `1 × 3 × 96 × 96` (CHW).

All four class probabilities are displayed live as bars plus a top-class badge.
The `Choose photo` button is a fallback when the camera is unavailable.

## Verification

`check_image.jpg` classified through the deployed page reproduces
`check_expected.json` (max absolute error < 3 × 10⁻⁶).

## QR code

The QR code at the bottom of the page is generated in the browser
(qrcode-generator, no external QR service) and encodes the deployed URL above.

## Files

| File | Purpose |
| --- | --- |
| `index.html` | the whole web app |
| `leaf_model.onnx` | the leaf classifier (ONNX, IR v10) |
| `model_info.json` | classes, input size, normalization |
| `check_image.jpg`, `check_expected.json` | correctness check |

[onnxruntime-web]: https://onnxruntime.ai/docs/tutorials/web/
