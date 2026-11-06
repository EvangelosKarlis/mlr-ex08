"""Task 4 -- the summary table from task4_results.jsonl.

    py -3.12 task4_summary.py        prints the table and writes task4_table.md

Accuracy is the mean over the seeds; the +- number is half the distance between
the best and the worst seed, so two rows differ only when the difference is
larger than their two ranges added together. Every row is the last epoch (a fair
comparison for all variants); the last column is the validation accuracy of the
best epoch of the run.
"""

import json

from task4 import VARIANTS, make_model

runs = [json.loads(line) for line in open("task4_results.jsonl", encoding="utf-8")]

by_variant = {}
for run in runs:
    by_variant.setdefault(run["variant"], []).append(run)


def mean(group, *keys):
    """The mean over the runs of one variant of runs[0][keys[0]][keys[1]]..."""
    values = group
    for key in keys:
        values = [value[key] for value in values]
    return sum(values) / len(values)


def half_range(group, *keys):
    """Half the distance between the best and the worst seed of the variant."""
    values = sorted(mean([run], *keys) for run in group)
    return (values[-1] - values[0]) / 2


def parameters_of(name):
    parameters = sum(p.numel() for p in make_model(
        next(variant for variant in VARIANTS if variant["name"] == name)).parameters())
    return f"{parameters / 1e6:.2f} M" if parameters > 1e5 else f"{parameters / 1e3:.0f} k"


def accuracy(group, *keys):
    return f"{mean(group, *keys):.3f} ±{half_range(group, *keys):.3f}"


lines = ["| variant | change vs the baseline | parameters | train acc | train loss "
         "| validation | camera-like | last year (OOD) | best epoch, val |",
         "|" + "|".join(["---"] * 9) + "|"]

for variant in VARIANTS:
    name = variant["name"]
    if name not in by_variant:
        continue
    group = by_variant[name]
    lines.append("| " + " | ".join([
        f"`{name}`",
        variant["comment"],
        parameters_of(name),
        f"{mean(group, 'train', 'accuracy'):.3f}",
        f"{mean(group, 'train', 'loss'):.3f}",
        accuracy(group, "last", "val", "accuracy"),
        accuracy(group, "last", "camera", "accuracy"),
        accuracy(group, "last", "history", "accuracy"),
        f"{mean(group, 'best_epoch_val_accuracy'):.3f}",
    ]) + " |")

table = "\n".join(lines)
print(table)
print(f"\n{sum(len(group) for group in by_variant.values())} runs, "
      f"seeds {sorted({run['seed'] for run in runs})}")

with open("task4_table.md", "w", encoding="utf-8") as handle:
    handle.write(table + "\n")
