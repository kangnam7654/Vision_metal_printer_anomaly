import sys
from pathlib import Path

ROOT_DIR = str(Path(__file__).parents[2])
sys.path.append("ROOT_DIR")

import cv2
import torch

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, files, transform=None, device=None):
        if device is None:
            self.device = 'cpu'
        else:
            self.device = device
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]  # albumentations transform
        image = image / 255
        return image.to(self.device)


if __name__ == "__main__":
    pass
