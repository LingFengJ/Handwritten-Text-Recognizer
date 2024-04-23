from sklearn.model_selection import train_test_split
import os
import torch
import numpy as np

import cv2

# Import dataset and dataloaders related packages

# from torchvision import datasets
# from torchvision.transforms import ToTensor
from torchvision.transforms import Compose, Grayscale

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from preprocessor import Preprocessor

class HandwritingDataset(Dataset):
    def __init__(self, data, vocabulary="", max_len = 0, transform=None, augmentations=False):
        self.data = data
        self.transform = transform
        self.vocab = vocabulary
        self.max_len = max_len
        self.data_preprocessor = Preprocessor(image=None, vocab=self.vocab, augmentation=augmentations)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # ------------------------------------------------------
        # preprocess the image
        # print(f'before resizing img.size(): {image.shape}')
        image, label_indices = self.data_preprocessor(image, label, self.max_len)

        if self.transform:
            image = self.transform(image)
        
        # disable these lines for visualizing the image in train.py
        image = torch.from_numpy(image).float()  # make it a tensor to enable it to be used by GPU
        image = image / 255.0  # Normalize pixel values to [0, 1]

        return image, label_indices

