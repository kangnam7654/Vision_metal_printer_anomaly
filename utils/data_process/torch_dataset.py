import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.common.project_paths import GetProjectPath
import os
import PIL
import torchvision.transforms as transforms
import glob
import numpy as np
import random
import cv2


def glob_files(folder_name='new_normal', load_file_num=None, sort=None):
    '''

    :param folder_name:
    :param load_file_num:
    :param sort:
    :return:
    '''
    paths = GetProjectPath()
    glob_path = os.path.join(paths.root_dir, 'data', folder_name, '*.png')
    file_list = glob.glob(glob_path)
    if load_file_num is None:
        return file_list
    else:
        num = load_file_num
        return file_list[:num]

class CustomDataset(Dataset):
    def __init__(self, file_list, transform=None, device=None):
        self.paths = GetProjectPath()
        if not device:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = f'cuda:{device}'
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = cv2.imread(img_path)
        image = cv2.resize(image, [224, 224])
        image = np.transpose(image, axes=[2, 0, 1])
        image = torch.tensor(image).float().to(self.device) / 255

        if self.transform:
            image = self.transform(image)
        return image



if __name__ == "__main__":
    pass