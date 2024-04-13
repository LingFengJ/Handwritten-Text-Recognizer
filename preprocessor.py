import random
from typing import Tuple

import cv2
import numpy as np



class Preprocessor:
    def __init__(self, image: np.ndarray, transform=None, augmentation=False, vocab=""):
        self.image = image
        self.transform = transform
        self.image_size = (256, 32) 
        # self.image_size = (1408, 96) ## an alternative for the image size(we'll see based on training results)
        self.augment = augmentation
        self.vocab = vocab
    
    def __call__(self, img:str, label:str, max_len : int = 32):
        # img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        img, label = self.preprocess_img(img, label)
        label = self.label_indexer(self.vocab, label)
        #label = self.label_padding(0, 32, label)
        label = self.label_padding(len(self.vocab), max_len, label)
        if self.augment:
            pass  # here we should apply randomsharpening, (randomnoise), randomblur, randombrightness

        return img, label

    @staticmethod
    def _truncate_label(text: str, max_text_len: int) -> str:
        """
        Function ctc_loss can't compute loss if it cannot find a mapping between text label and input
        labels. Repeat letters cost double because of the blank symbol needing to be inserted.
        If a too-long label is provided, ctc_loss returns an infinite gradient.
        """
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i - 1]:
                cost += 2
            else:
                cost += 1
            if cost > max_text_len:
                return text[:i]
        return text

    def preprocess_img(self, img: np.ndarray, text: str) -> Tuple[np.ndarray, str]:
        """
        Resize image and truncate text label if it is too long
        """
        # # Resize image
        # img = cv2.resize(img, self.image_size)

        # # Truncate text label
        # text = self._truncate_label(text, max_text_len=32)

        wt, ht = self.image_size
        h, w = img.shape
        f = min(wt / w, ht / h)
        tx = (wt - w * f) / 2
        ty = (ht - h * f) / 2

        # map image into target image
        M = np.float32([[f, 0, tx], [0, f, ty]])
        target = np.ones([ht, wt]) * 255
        img = cv2.warpAffine(img, M, dsize=(wt, ht), dst=target, borderMode=cv2.BORDER_TRANSPARENT)
        
        return img, text

    @staticmethod
    def label_indexer(vocab: str, label: np.ndarray):
        """Convert label to index by vocab
            vocab (typing.List[str]): List of characters in vocab
        """
        # def __call__(self, data: np.ndarray, label: np.ndarray):
        return np.array([vocab.index(l) for l in label if l in vocab])

    @staticmethod
    def label_padding(padding_value: int, max_len: int, label: np.ndarray):
        label = label[:max_len]
        # put 0 padding values on the left, max_len - len(labels) padding values on the right
        return np.pad(label, (0,max_len - len(label)), mode='constant', constant_values=padding_value)
