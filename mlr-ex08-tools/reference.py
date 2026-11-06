"""Reference preprocessing + inference to match check_expected.json.

Usage: py -I reference.py <model.onnx> <check_image.jpg> <model_info.json> <expected.json>
"""
import json
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image


def classify(img, info, sess, inp_name, size, resample):
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    crop = img.crop((left, top, left + side, top + side))
    small = crop.resize((size, size), resample)
    arr = np.asarray(small, dtype=np.float32)[:, :, :3] / 255.0
    mean = np.array(info["mean"], dtype=np.float32)
    std = np.array(info["std"], dtype=np.float32)
    arr = (arr - mean) / std
    chw = np.transpose(arr, (2, 0, 1))[None].astype(np.float32)
    out = sess.run(None, {inp_name: chw})[0][0]
    if out.min() < 0 or out.sum() < 0.5 or out.sum() > 1.5:
        e = np.exp(out - out.max())
        out = e / e.sum()
    return {c: float(p) for c, p in zip(info["classes"], out)}


def main():
    model_path, img_path, info_path, exp_path = sys.argv[1:5]
    info = json.load(open(info_path))
    expected = json.load(open(exp_path))
    size = int(info["input_size"])
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    print("input:", inp_name, sess.get_inputs()[0].shape,
          "output:", sess.get_outputs()[0].name, sess.get_outputs()[0].shape)

    im = Image.open(img_path).convert("RGB")
    print("check_image size:", im.size)

    filters = {
        "BILINEAR": Image.Resampling.BILINEAR,
        "BICUBIC": Image.Resampling.BICUBIC,
        "NEAREST": Image.Resampling.NEAREST,
        "LANCZOS": Image.Resampling.LANCZOS,
        "BOX": Image.Resampling.BOX,
        "HAMMING": Image.Resampling.HAMMING,
    }
    for name, flt in filters.items():
        probs = classify(im, info, sess, inp_name, size, flt)
        err = max(abs(probs[k] - expected[k]) for k in expected)
        print("%-9s maxabs_err=%.3e  %s" % (
            name, err, {k: round(v, 6) for k, v in probs.items()}))


main()
