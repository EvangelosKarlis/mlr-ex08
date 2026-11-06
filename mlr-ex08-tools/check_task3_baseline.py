"""Check that task4.LeafNetBN() with its defaults is exactly the Task 3d network.

The Task 4 baseline has to be the Task 3 network, so this defines the class as
it is written in ex08_student.ipynb cell 60 and compares it, built under the
same seed, with the one task4.py actually trains.

    .venv-task4/Scripts/python.exe mlr-ex08-tools/check_task3_baseline.py
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
import task4


class LeafNetBN_Task3(nn.Module):
    """Verbatim from the notebook, cell 60."""

    def __init__(self):
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
        self.fc1 = nn.Linear(64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)


torch.manual_seed(0)
task3 = LeafNetBN_Task3()
torch.manual_seed(0)
mine = task4.LeafNetBN()

keys_task3, keys_mine = list(task3.state_dict()), list(mine.state_dict())
print("same tensors in the same order:", keys_task3 == keys_mine)
print("parameters:", sum(p.numel() for p in task3.parameters()),
      "vs", sum(p.numel() for p in mine.parameters()))
same_values = all(torch.equal(a, b) for a, b in zip(task3.state_dict().values(),
                                                    mine.state_dict().values()))
print("every weight and bias identical:", same_values)

x = torch.zeros(2, 3, 96, 96)
task3.eval()
mine.eval()
with torch.no_grad():
    same_output = torch.equal(task3(x), mine(x))
print("same output on the same input:", same_output)
print("OK" if (keys_task3 == keys_mine and same_values and same_output) else "DIFFERENT")
