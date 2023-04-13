import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_transform(size=(512, 512)):
    seq = [A.Resize(*size), ToTensorV2()]
    transform = A.Compose(seq)
    return transform