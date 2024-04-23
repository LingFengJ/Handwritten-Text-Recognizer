from sklearn.model_selection import train_test_split
import os
import torch
import numpy as np
from datetime import datetime

import cv2
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Grayscale
from torch.utils.data import DataLoader
from tqdm import tqdm

from preprocessor import Preprocessor
from data_loader import HandwritingDataset





# import various models to test
from model1 import Model as Model1
# from model2 import Network as Model2
# from crnn import CRNN

import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.nn import CTCLoss
# from early_stopping import EarlyStopping
# from model_checkpoint import ModelCheckpoint
# from train_logger import TrainLogger
# from reduce_lr_on_plateau import ReduceLROnPlateau
# from model2onnx import Model2onnx
# from metric_plus_callback import WERMetric, CERMetric, EarlyStopping, ModelCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau

def set_device():
    """
    Set the device. CUDA if available, CPU otherwise

    Args:
    None

    Returns:
    Nothing
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("GPU is not enabled in this notebook. \n"
                "If you want to enable it, in the menu under `Runtime` -> \n"
                "`Hardware accelerator.` and select `GPU` from the dropdown menu")
    else:
        print("GPU is enabled in this notebook. \n"
                "If you want to disable it, in the menu under `Runtime` -> \n"
                "`Hardware accelerator.` and select `None` from the dropdown menu")

    return device



def main():
    device = set_device()
    # Set the seed
    seed = 2024
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    sentences_txt_path = os.path.join('data', 'iam_sentences', 'metadata', 'sentences.txt')
    sentences_folder_path = os.path.join('data', 'iam_sentences', 'dataset')

    dataset, vocab, max_len = [], set(), 0
    words = open(sentences_txt_path, "r").readlines()
    for line in tqdm(words):
        if line.startswith("#"):
            continue

        line_split = line.split(" ")
        if line_split[2] == "err":
            continue

        # folder1 = line_split[0][:3]
        # folder2 = line_split[0][:8]
        file_name = line_split[0] + ".png"
        label = line_split[-1].rstrip('\n')

        # recplace '|' with ' ' in label
        label = label.replace('|', ' ')

        rel_path = os.path.join(sentences_folder_path, file_name)  # stow.join()
        if not os.path.exists(rel_path):  #stow.exits()
            continue

        dataset.append([rel_path, label])
        vocab.update(list(label))
        max_len = max(max_len, len(label))

vocab = ''.join(vocab)  # vocabulary: a string containing all possible characters

def classfold(): 
    # -------------------------------------dataset class------------------------------------

    # class HandwritingDataset(Dataset):
    #     def __init__(self, data, vocabulary="", transform=None):
    #         self.data = data
    #         self.transform = transform
    #         self.vocab = vocabulary

    #     def __len__(self):
    #         return len(self.data)

    #     def __getitem__(self, idx):
    #         image_path, label = self.data[idx]
    #         image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #         # image = torch.from_numpy(image).float()

    #         # # Resize image to have the longest side be 224
    #         # h, w = image.shape
    #         # if h > w:
    #         #     new_h, new_w = 224, int(224 * w / h)
    #         # else:
    #         #     new_h, new_w = int(224 * h / w), 224
    #         # image = cv2.resize(image, (new_w, new_h))

    #         # # # Pad the image to 224x224
    #         # pad_h = 224 - new_h
    #         # pad_w = 224 - new_w
    #         # image = np.pad(image, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)), mode='constant')

    #         # image = image / 255.0  # Normalize pixel values to [0, 1]
    #         # image = image[np.newaxis, :]  # Add a channel dimension
    #         # image = torch.from_numpy(image).float()  # Convert to torch tensor
    #         # if self.transform:
    #         #     image = self.transform(image)

    #         # image = torch.from_numpy(image).float()
    #         # ------------------------------------------------------
    #         # preprocess the image
    #         #wt, ht = (256, 32)
    #         wt, ht = (1408, 96)
    #         h, w = image.shape
    #         f = min(wt / w, ht / h)
    #         tx = (wt - w * f) / 2
    #         ty = (ht - h * f) / 2

    #         # map image into target image
    #         M = np.float32([[f, 0, tx], [0, f, ty]])
    #         target = np.ones([ht, wt]) * 255
    #         image = cv2.warpAffine(image, M, dsize=(wt, ht), dst=target, borderMode=cv2.BORDER_TRANSPARENT)

    #         # Apply LabelIndexer function to convert label to indices (in order to have numerical labels)
    #         label_indices = Preprocessor.label_indexer(self.vocab, label)
    #         print('label:', label, 'with len of:', len(label))
    #         print('label indices:', label_indices, 'with len of:', len(label_indices))

    #         # Apply LabelPadding function to pad the labels to the maximum length (in order to have fixed length labels)
    #         label_indices = Preprocessor.label_padding(len(self.vocab), max_len, label_indices)
    #         print('vocab:', self.vocab, '---with len of:', len(self.vocab))
    #         print('padded labels:', label_indices)

    #         if self.transform:
    #             image = self.transform(image)

    #         return image, label_indices
    pass


# Split the dataset into training, validation, and test sets
train_data, test_data = train_test_split(dataset, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

# Create dataset objects
train_dataset = HandwritingDataset(train_data, vocab, max_len, augument=True)
val_dataset = HandwritingDataset(val_data, vocab, max_len)
test_dataset = HandwritingDataset(test_data, vocab, max_len)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print("len of loaders:",train_loader.__len__(), val_loader.__len__(), test_loader.__len__())

def testing_images():
#--------------------------testing--------------------------------

    import matplotlib.pyplot as plt

    print(train_loader.__len__(), val_loader.__len__(), test_loader.__len__())
    # Get one batch of images from the train_loader
    images, temp_labels = next(iter(train_loader))

    # # Plot the first image from the batch
    # plt.imshow(images[0].squeeze(), cmap='gray')
    # plt.show()

    # Plot the first 5 images from the batch
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    for i in range(3):
        for j in range(2):
        # print(images[i].shape)
            axs[i,j].imshow(images[i*3+j].squeeze(), cmap='gray') #squeeze to remove channel dimension
        # axs[i].imshow(images[i].permute(1, 2, 0), cmap='gray') # if the image is rgb
            print(images[i].shape, temp_labels[i])
    plt.show()