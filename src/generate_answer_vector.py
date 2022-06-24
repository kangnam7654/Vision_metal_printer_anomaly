import os
from pathlib import Path

import glob
import pickle
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.feature_extractor.load_feature_extractor import load_feature_extractor
from utils.data_process.transform import get_transform
from utils.data_process.torch_dataset import CustomDataset


ROOT_DIR = Path(__file__).parent
DATA_DIR = os.path.join(ROOT_DIR, "data", "normal")
PICK_DIR = os.path.join(ROOT_DIR, "pickle")


def generate_answer_vector():
    device = "cuda"
    data_path = os.path.join(DATA_DIR, "*.png")
    files = glob.glob(data_path, recursive=True)
    size = (224, 224)
    transform = get_transform(size=size)

    data_set = CustomDataset(files=files, transform=transform, device=device)
    dataloader = DataLoader(data_set, shuffle=False, batch_size=1, drop_last=False)

    feature_extractor = load_feature_extractor(device=device)

    tmp = []
    for data in tqdm(dataloader):
        output = feature_extractor(data)
        tmp.append(output.cpu().detach().numpy())

    stacked = np.stack(tmp, axis=0)
    vector = np.mean(stacked, axis=0).squeeze(0)  # (1, 512, 7, 7) -> (512, 7, 7)
    with open(os.path.join(PICK_DIR, "answer_vecs.pkl"), "wb") as f:
        pickle.dump(vector, f)


if __name__ == "__main__":
    generate_answer_vector()
    