# data_loader.py
import pandas as pd
import torch, nibabel as nib
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from config import Config

class MedicalDataset(Dataset):
    def __init__(self, csv_path, modality, augment=False):
        self.df   = pd.read_csv(csv_path)
        self.mod  = modality
        self.aug  = augment
        self.tf   = self._make_transforms()

    def _make_transforms(self):
        if self.mod in ['ct','mri']:
            from monai.transforms import Compose, RandFlip, RandRotate
            t = [RandFlip(spatial_axis=0, prob=0.5),
                 RandRotate(range_x=15, prob=0.5)]
            return Compose(t)
        t = [transforms.Resize(Config.IMAGE_SIZE_2D),
             transforms.ToTensor()]
        if self.aug:
            t.insert(1, transforms.RandomRotation(10))
            t.insert(2, transforms.RandomHorizontalFlip())
        return transforms.Compose(t)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path  = row['path']
        label = row[f"{self.mod}_label"]
        if self.mod in ['ct','mri']:
            vol  = nib.load(path).get_fdata().astype('float32')
            img  = torch.tensor(vol).unsqueeze(0)
            img  = self.tf(img)
        else:
            img  = Image.open(path).convert('RGB')
            img  = self.tf(img)
        return img, torch.tensor(label)

def get_loader(csv_path, modality, batch, shuffle=True, augment=False):
    ds = MedicalDataset(csv_path, modality, augment=augment)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=4)


