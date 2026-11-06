"""Task 4 -- a better leaf classifier, one change at a time.

Everything below the "Task 3 pipeline" marker is copied from ex08_student.ipynb
(cells 40, 54, 56, 58, 60, 61). The Task 4 section adds one small switch per
variant, so every variant differs from BASELINE in exactly one line of the
configuration table VARIANTS at the bottom.

    py -3.12 task4.py <variant> [seed]      one run, e.g. task4.py v1-color 42
    py -3.12 task4.py --list                the variant names

Every run appends one JSON line to task4_results.jsonl with the accuracy of the
last epoch and of the best epoch on three test sets: our validation set, the
fixed-seed camera-like set, and last year's photographs.
"""

import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Task 3 pipeline (ex08_student.ipynb) -- unchanged except for the folder names
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

TRAIN_FOLDER = "ex08_leaves_group1/leaves/train"
VAL_FOLDER = "ex08_leaves_group1/leaves/val"
HIST_TRAIN_FOLDER = "data_history/leaves/train"
HIST_VAL_FOLDER = "data_history/leaves/val"


class LeafDataset(Dataset):
    """The Dataset of Task 3a, unchanged."""

    def __init__(self, folder, transform):
        self.transform = transform
        # The class names: the sorted subfolders of the folder.
        # A list of (file path, class index) pairs: every class, its files in sorted order.
        self.classes = sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])
        self.samples = []
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(folder, class_name)
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    self.samples.append((os.path.join(class_dir, fname), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

# The geometric augmentation of Task 3d ("the two improvements").
aug_transform = transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])


class LeafNet(nn.Module):
    """The network of Task 3b, unchanged. Only used for the reference row."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)


class LeafNetBN(nn.Module):
    """The network of Task 3d, unchanged when called without arguments.

    Task 4 adds two optional switches, both off by default:
      dropout      -- 0.0 means no dropout, exactly as in Task 3;
      global_pool  -- False keeps the flatten of Task 3, True replaces it by
                      nn.AdaptiveAvgPool2d((1, 1)).
    """

    def __init__(self, dropout=0.0, global_pool=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) if global_pool else None
        self.fc1 = nn.Linear(64 if global_pool else 64 * 6 * 6, 64)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.global_pool(x).flatten(1) if self.global_pool is not None else x.flatten(1)
        x = F.relu(self.fc1(x))
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Task 4 additions
# ---------------------------------------------------------------------------

# What a phone camera does to the same leaf: it is not centred, not upright and
# not lit the same way. Used only to BUILD the camera-like validation set.
camera_transform = transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.4, 1.0), ratio=(0.7, 1.4)),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

CAMERA_SEED, CAMERA_VIEWS = 1234, 8


def build_camera_val_set(folder):
    """Every validation photograph seen CAMERA_VIEWS times through the camera
    transform, generated ONCE with a fixed seed (CAMERA_SEED + the index of the
    image) so that all variants are measured on exactly the same pictures.

    Seeding the random crops also moves the training run's random stream, so we
    put it back afterwards: without these two lines every seed would train on
    the same initial weights and the same batch order.
    """
    saved_state = torch.get_rng_state()
    dataset = LeafDataset(folder, transform=None)
    views, labels = [], []
    for i, (path, label) in enumerate(dataset.samples):
        image = Image.open(path).convert("RGB")
        torch.manual_seed(CAMERA_SEED + i)
        for _ in range(CAMERA_VIEWS):
            views.append(camera_transform(image))
            labels.append(label)
    torch.set_rng_state(saved_state)
    return TensorDataset(torch.stack(views), torch.tensor(labels))


# The colour augmentation of Task 4: the geometric one of Task 3d plus ColorJitter.
aug_color_transform = transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])


def make_dataset(folder, augmentation):
    return LeafDataset(folder, aug_color_transform if augmentation == "color"
                       else aug_transform if augmentation == "geometry"
                       else transform)


def make_loader(dataset, shuffle):
    return DataLoader(dataset, batch_size=32, shuffle=shuffle)


def make_model(variant):
    """One branch per model in the table below; nothing else builds a model."""
    name = variant["model"]
    if name == "leafnet":
        return LeafNet()
    if name == "leafnetbn":
        return LeafNetBN(dropout=variant.get("dropout", 0.0),
                         global_pool=variant.get("global_pool", False))
    if name == "mobilenet":
        model = models.mobilenet_v3_small(weights="DEFAULT")   # the Task 2 network
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
        return model
    raise ValueError(name)


def make_optimizer(variant, parameters):
    if variant.get("optimizer", "Adam") == "AdamW":
        return torch.optim.AdamW(parameters, lr=variant["lr"], weight_decay=variant["weight_decay"])
    return torch.optim.Adam(parameters, lr=variant["lr"])


def evaluate(model, loader):
    """Accuracy and mean loss of the model on a loader (evaluation mode, no gradients)."""
    model.eval()
    correct, loss_sum, total = 0, 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_sum += F.cross_entropy(outputs, labels).item() * len(labels)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += len(labels)
    return correct / total, loss_sum / total


def measure(model, test_sets):
    return {name: evaluate(model, loader) for name, loader in test_sets.items()}


def train(variant, seed):
    """The training loop of Task 3c/3d, with four Task 4 additions marked
    [T4]: train accuracy, an optional scheduler, the weights of the best epoch,
    and the three test sets at the end."""
    name, epochs = variant["name"], variant["epochs"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # The three test sets, identical for every variant and every seed.
    val_loader = make_loader(make_dataset(VAL_FOLDER, "none"), shuffle=False)
    camera_loader = DataLoader(build_camera_val_set(VAL_FOLDER), batch_size=32)
    history_loader = make_loader(make_dataset(HIST_VAL_FOLDER, "none"), shuffle=False)
    train_loader = make_loader(make_dataset(TRAIN_FOLDER, variant["augmentation"]), shuffle=True)
    if variant.get("add_history"):
        train_loader = make_loader(
            ConcatDataset([make_dataset(TRAIN_FOLDER, variant["augmentation"]),
                           make_dataset(HIST_TRAIN_FOLDER, variant["augmentation"])]), shuffle=True)

    model = make_model(variant).to(device)
    if variant.get("freeze_backbone"):                     # [Task 4] frozen transfer
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]

    criterion = nn.CrossEntropyLoss(label_smoothing=variant.get("label_smoothing", 0.0))
    optimizer = make_optimizer(variant, trainable)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                 if variant.get("scheduler") else None)     # [T4] cosine schedule

    best_accuracy, best_weights = -1.0, None
    started = time.time()
    for epoch in range(epochs):
        model.train()
        if variant.get("freeze_backbone"):
            model.features.eval()          # [T4] frozen backbone: keep its BN statistics too
        loss_sum, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            # [T4] the training accuracy, from the forward pass we already have:
            # a second one would update the batch normalization statistics twice.
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += len(labels)

        val_accuracy, _ = evaluate(model, val_loader)
        if val_accuracy > best_accuracy:                   # [T4] keep the best epoch
            best_accuracy, best_weights = val_accuracy, copy.deepcopy(model.state_dict())
        if scheduler is not None:
            scheduler.step()

        print(f"epoch {epoch + 1:2d}/{epochs}  train loss {loss_sum / total:.4f} "
              f"train acc {correct / total:.3f}  val acc {val_accuracy:.3f}  "
              f"lr {optimizer.param_groups[0]['lr']:.5f}", flush=True)

    def report(model_state):
        model.load_state_dict(model_state)
        scores = measure(model, {"val": val_loader, "camera": camera_loader, "history": history_loader})
        return {key: {"accuracy": value[0], "loss": value[1]} for key, value in scores.items()}

    last_weights = copy.deepcopy(model.state_dict())
    results = {
        "variant": name, "seed": seed, "epochs": epochs, "seconds": round(time.time() - started, 1),
        "train": {"loss": loss_sum / total, "accuracy": correct / total},   # last epoch
        "last": report(last_weights),
        "best": report(best_weights),
        "best_epoch_val_accuracy": best_accuracy,
    }
    with open("task4_results.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(results) + "\n")
    print(f"{name} seed={seed}: val {results['last']['val']['accuracy']:.3f} "
          f"camera {results['last']['camera']['accuracy']:.3f} "
          f"history {results['last']['history']['accuracy']:.3f} "
          f"(best epoch val {best_accuracy:.3f}) in {results['seconds']}s", flush=True)
    return results


# ---------------------------------------------------------------------------
# The variants. Each one is the baseline plus ONE change, named in the comment.
# ---------------------------------------------------------------------------

VARIANTS = [
    # the baseline: exactly the Task 3d pipeline (LeafNetBN, geometric augmentation,
    # Adam 1e-3, 15 epochs, cross entropy without smoothing, flatten)
    dict(name="baseline", model="leafnetbn", augmentation="geometry", epochs=15, lr=1e-3,
         comment="Task 3d network: the fixed baseline"),
    dict(name="ref-plain", model="leafnetbn", augmentation="none", epochs=15, lr=1e-3,
         comment="baseline WITHOUT augmentation (context for the baseline row)"),
    dict(name="v1-color", model="leafnetbn", augmentation="color", epochs=15, lr=1e-3,
         comment="ONE change: ColorJitter added to the augmentation"),

    # transfer learning: the Task 2 MobileNet, head only vs fine-tuned
    dict(name="v2a-frozen", model="mobilenet", augmentation="geometry", epochs=15, lr=1e-3,
         freeze_backbone=True, comment="ONE change: MobileNet with only the new last layer trained"),
    dict(name="v2b-finetune", model="mobilenet", augmentation="geometry", epochs=15, lr=1e-3,
         comment="ONE change: MobileNet, whole network fine-tuned"),
    dict(name="v2b-noaug", model="mobilenet", augmentation="none", epochs=15, lr=1e-3,
         comment="v2b without augmentation = the recipe of leaf_model.onnx, the model now on the page"),
    dict(name="v3-schedule", model="mobilenet", augmentation="color", epochs=40, lr=1e-3,
         scheduler=True,
         comment="v2b + colour augmentation + CosineAnnealingLR + 40 epochs"),

    # regularization / optimization: one change each, then together
    dict(name="v4a-dropout", model="leafnetbn", augmentation="geometry", epochs=15, lr=1e-3, dropout=0.5,
         comment="ONE change: Dropout(0.5) before the last layer"),
    dict(name="v4b-gap", model="leafnetbn", augmentation="geometry", epochs=15, lr=1e-3, global_pool=True,
         comment="ONE change: global average pooling instead of flatten"),
    dict(name="v4c-adamw", model="leafnetbn", augmentation="geometry", epochs=15, lr=1e-3,
         optimizer="AdamW", weight_decay=1e-4,
         comment="ONE change: AdamW with weight decay 1e-4"),
    dict(name="v4d-smooth", model="leafnetbn", augmentation="geometry", epochs=15, lr=1e-3,
         label_smoothing=0.1, comment="ONE change: label smoothing 0.1"),
    dict(name="v4e-combined", model="leafnetbn", augmentation="geometry", epochs=15, lr=1e-3,
         dropout=0.5, global_pool=True, optimizer="AdamW", weight_decay=1e-4, label_smoothing=0.1,
         comment="v4a+b+c+d together, and the weights of the best epoch are the ones kept"),

    # last year's photographs: more training data, other students and phones
    dict(name="v5-history", model="leafnetbn", augmentation="geometry", epochs=15, lr=1e-3,
         add_history=True, comment="ONE change: baseline trained on our 101 + 700 historical images"),
]


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if not arguments or arguments[0] == "--list":
        for variant in VARIANTS:
            print(f"{variant['name']:14s} {variant['comment']}")
        sys.exit(0)
    wanted, seed = arguments[0], int(arguments[1]) if len(arguments) > 1 else 42
    chosen = next((v for v in VARIANTS if v["name"] == wanted), None)
    if chosen is None:
        sys.exit(f"unknown variant {wanted!r}, try --list")
    train(chosen, seed)
