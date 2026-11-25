# -*- coding: utf-8 -*-
from torch.utils.data import Dataset, DataLoader
from PIL import Image as Image
import os
from torchvision.transforms import functional
#Dataset
class CorrectionDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'src/'))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        src = Image.open(os.path.join(self.image_dir, 'src', self.image_list[idx]))
        correction = Image.open(os.path.join(self.image_dir, 'GTC', self.image_list[idx]))
        distortion = Image.open(os.path.join(self.image_dir, 'GTF', self.image_list[idx]))
        
        src = functional.to_tensor(src)
        correction = functional.to_tensor(correction)
        distortion = functional.to_tensor(distortion)
        
        name = self.image_list[idx]
        return src,correction,distortion,name

#DataLoader
def train_dataloader(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path, 'train')

    dataloader = DataLoader(
        CorrectionDataset(image_dir),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path, 'test')
    image_dir = path
    dataloader = DataLoader(
        CorrectionDataset(image_dir),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader
