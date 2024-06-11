import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
import numpy as np
from glob import glob

class Dataset(Dataset):
    def __init__(self, df, fold, mode, img_size):
        self.mode = mode
        if self.mode == 'train':
            self.df = df[df['fold'] != fold].reset_index(drop=True)
        elif self.mode == 'valid':
            self.df = df[df['fold'] == fold].reset_index(drop=True)
        else:
            self.df = df

        self.images = self.df['image_path'].tolist()
        self.labels = self.df['label'].tolist()
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.img_size = img_size
        self.transforms = self.get_transforms()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        image = np.asarray(Image.open(self.images[index]).convert('RGB'), dtype=np.float32)
        image = self.transforms(image=image)['image']
        image = self.norm(image)
        image = torch.from_numpy(image.transpose((2,0,1)))
        label = torch.as_tensor(self.labels[index])
        return image, label
    
    def norm(self, img):
        img = img.astype(np.float32)
        img = img/255.
        img -= self.mean
        img *= np.reciprocal(self.std, dtype=np.float32)
        return img
    
    def get_transforms(self,):
        if self.mode == 'train':
            transforms=(A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.5),
            ]))
        else:
            transforms=(A.Compose([A.Resize(self.img_size, self.img_size)]))
        return transforms