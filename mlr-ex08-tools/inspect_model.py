import sys, json
import onnxruntime as ort

model_path = sys.argv[1]
sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
for inp in sess.get_inputs():
    print("input:", inp.name, inp.type, inp.shape)
for out in sess.get_outputs():
    print("output:", out.name, out.type, out.shape)
