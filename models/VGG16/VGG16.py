import numpy as np
import torchvision.models as models
from utils.data_process import torch_dataset
import pickle
from torch.utils.data import DataLoader
import torch
import copy

def model_freeze(model):
    for param in model.parameters():
        param.requires_grad_(False)
    return model.eval()


def load_vgg16(freeze_sw=True, device=None):
    import torch
    if not device:
        device_set = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device_set = f'cuda:{device}'
    vgg16 = models.vgg16(pretrained=True)

    if freeze_sw:
        vgg16 = model_freeze(vgg16)
    return vgg16.to(device_set)


def generate_feature_vectors(load_file_num=None):
    files_list = torch_dataset.glob_files('new_normal', load_file_num=load_file_num)
    data_pre = torch_dataset.CustomDataset(files_list)
    dataloader = DataLoader(data_pre, shuffle=False, batch_size=1, drop_last=False)

    model = load_vgg16()
    model_freeze(model)

    inferences_list = []

    for idx, data in enumerate(dataloader):
        output = model(data.float())
        inferences_list.append(output)
        print(f'[{idx+1}/{len(dataloader)}] completed')

        if idx % 10 == 9:
            ckpt_copy = copy.deepcopy(inferences_list)
            ckpt_cat = torch.cat(ckpt_copy, dim=0)
            ckpt_mean = torch.mean(ckpt_cat)
            with open('/Users/kimkangnam/Desktop/Project/CompanyProject/DataVoucher/Hongsworks/pickle/answer_vectors_2.pkl', 'wb') as f:
                pickle.dump(ckpt_mean, f)
            print(f'check point saved! {idx+1}')
    inferences_cat = torch.cat(inferences_list, dim=0)
    inferences_mean = torch.mean(inferences_cat, dim=0)
    with open('/Users/kimkangnam/Desktop/Project/CompanyProject/DataVoucher/Hongsworks/pickle/answer_vectors_2.pkl',
              'wb') as f:
        pickle.dump(inferences_mean, f)

if __name__ == '__main__':
    generate_feature_vectors()