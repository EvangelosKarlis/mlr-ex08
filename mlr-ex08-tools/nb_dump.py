"""Dump notebook cell sources/outputs. Run: py -3.12 mlr-ex08-tools/nb_dump.py <nb> <idx...>"""
import json
import sys

nb = json.load(open(sys.argv[1], encoding="utf-8"))
for i in (int(a) for a in sys.argv[2:]):
    cell = nb["cells"][i]
    print("=" * 20, "CELL", i, cell["cell_type"], "=" * 20)
    print("".join(cell["source"]))
    for o in cell.get("outputs", []):
        text = o.get("text") or o.get("data", {}).get("text/plain")
        if text:
            print("--- output ---")
            print("".join(text))
