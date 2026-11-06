"""Minimal protobuf peek at an ONNX ModelProto: ir_version, opset_imports, producer."""
import sys

def read_varint(buf, i):
    shift = 0
    val = 0
    while True:
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            return val, i
        shift += 7

def walk(buf, start, end, depth):
    i = start
    while i < end:
        key, i = read_varint(buf, i)
        field, wt = key >> 3, key & 7
        if wt == 0:
            val, i = read_varint(buf, i)
            yield depth, field, val
        elif wt == 2:
            ln, i = read_varint(buf, i)
            if depth == 0 and field in (2, 3, 4):  # producer_name, producer_version, doc_string
                yield depth, field, buf[i:i+ln].decode("utf-8", "replace")
            elif depth == 1 and field == 1:  # OperatorSetIdProto.domain
                yield depth, field, buf[i:i+ln].decode("utf-8", "replace") or "ai.onnx"
            else:
                yield depth, field, ("len=%d" % ln)
            i += ln
        elif wt == 5:
            i += 4
        elif wt == 1:
            i += 8
        else:
            raise ValueError("bad wiretype %d at %d" % (wt, i))

path = sys.argv[1]
with open(path, "rb") as f:
    buf = f.read()
print("bytes:", len(buf))
i = 0
while i < len(buf):
    key, j = read_varint(buf, i)
    field, wt = key >> 3, key & 7
    if field == 8 and wt == 2:  # opset_import submessage
        ln, k = read_varint(buf, j)
        for d, fld, v in walk(buf, k, k + ln, 1):
            print("opset_import field", fld, "=", v)
        i = k + ln
    elif wt == 0:
        val, i = read_varint(buf, j)
        print("top-level field", field, "=", val)
    elif wt == 2:
        ln, k = read_varint(buf, j)
        if field in (2, 3, 4):
            print("top-level field", field, "=", buf[k:k+ln].decode("utf-8", "replace"))
        i = k + ln
    else:
        break
    if field == 7:  # graph; stop, rest is weights
        break
