import random
from typing import Tuple

import cv2
import numpy as np
from torchvision import transforms as transforms
import typing
import torch



class Preprocessor:
    def __init__(self, image: np.ndarray, transform=None, augmentation=False, vocab=""):
        self.image = image
        self.transform = transform
        # self.image_size = (256, 32) 
        # self.image_size = (512, 64)
        # self.image_size = (128, 36)
        self.image_size = (400, 32)
        # self.image_size = (800, 64)
        # self.image_size = (1408, 96) ## an alternative for the image size(we'll see based on training results)
        self.augment = augmentation
        self.vocab = vocab
    
    def __call__(self, img, label:str, max_len : int = 32):
        # img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        img, label = self.preprocess_img(img, label)
        label = self.label_indexer(self.vocab, label)
        #label = self.label_padding(0, 32, label)
        label = self.label_padding(len(self.vocab), max_len, label)

        if self.augment:
            # kernel = np.ones((5,5),np.float32)/25
            # dst = cv.filter2D(img,-1,kernel)
            # kernel = np.ones((2,2),np.float32)/20
            # img = cv2.Laplacian(img, cv2.CV_16S, ksize=3)

            # here we should apply randomsharpening, (randomnoise), randomblur, randombrightness
            # if np.random.rand() < 0.25:
            #     # random_odd = np.random.randint(1,3) * 2 + 1
            #     random_deviation = np.random.uniform(0, 1.5)
            #     img = cv2.GaussianBlur(img, (0, 0), random_deviation)

            # if np.random.rand() < 0.25:
            #     # brightness = np.random.randint(0,50)
            #     # img = cv2.add(img, brightness)
            #     delta = 100
            #     if np.random.rand() < 0.5:
            #         value = 1 + np.random.uniform(-delta, delta) / 255
            #         # Convert grayscale image to BGR
            #         image_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            #         # Convert BGR image to HSV
            #         hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            #         # Duplicate single channel across all three channels
            #         hsv[..., 1] = hsv[..., 1] * value
            #         hsv[..., 2] = hsv[..., 2] * value
            #         # Clip values to the range [0, 255]
            #         hsv = np.clip(hsv, 0, 255)
            #         # Convert HSV image back to BGR
            #         img_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            #         # Convert BGR image to grayscale
            #         img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # if np.random.rand() < 0.25:
            #     # Convert the shear angle to radians
            #     ang_rad = np.deg2rad(5)

            #     # Create the shear transformation matrix
            #     M = np.array([[1, np.tan(ang_rad), 0],
            #                 [0, 1, 0]], dtype=np.float32)

            #     rows, cols = img.shape[:2]
            #     img = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        
            def random_sharpen(image: np.ndarray, 
                                random_chance: float = 0.5,
                                alpha: float = 0.25,
                                lightness_range: typing.Tuple[float, float] = (0.75, 2.0),
                                kernel: np.ndarray = None,
                                kernel_anchor: np.ndarray = None,
                                ) -> np.ndarray:
                if np.random.rand() < random_chance:
                    lightness = np.random.uniform(*lightness_range)
                    alpha = np.random.uniform(alpha, 1.0)

                    kernel = np.array([[-1, -1, -1], [-1,  1, -1], [-1, -1, -1]], dtype=np.float32) if kernel is None else kernel
                    kernel_anchor = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32) if kernel_anchor is None else kernel_anchor

                    kernel = kernel_anchor * (8 + lightness) + kernel
                    kernel -= kernel_anchor
                    kernel = (1 - alpha) * kernel_anchor + alpha * kernel

                    # Apply sharpening
                    sharpened_image = cv2.filter2D(image, -1, kernel)
                    return sharpened_image
                else:
                    return image
            img = random_sharpen(img)


            transformations = transforms.Compose(
            [
                transforms.ToPILImage(),
                # Other possible data augmentation
                # transforms.RandomAffine(degrees=(-3, 3), translate=(0, 0.2), scale=(0.9, 1),
                #                         shear=5, resample=False, fillcolor=255),
                transforms.RandomAffine(degrees=(-2, 2), translate=(0, 0), scale=(0.9, 1),
                                        shear=5,fill=255),
                # transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3, fill=255)
            ]
            )
            if np.random.rand() < 0.5:
                img = np.array(transformations(img))
            # print(f'final image size: {img.shape}')

            # if np.random.rand() < 0.25:
            #     random_odd = np.random.randint(1,3) * 2 + 1 # a random odd number between 3 and 5
            #     kernel = np.ones((random_odd,random_odd),np.uint8)
            #     img = cv2.erode(img, kernel, iterations = 1)
            # if np.random.rand() < 0.25:
            #     sharpen_factor = 1 + np.random.rand()
            #     kernel = np.array([[0, -1, 0], 
            #                [-1, sharpen_factor,-1], 
            #                [0, -1, 0]], dtype='float32')
            #     img = cv2.filter2D(img, -1, kernel)

            # if np.random.rand() < 0.25:            
            #     img = cv2.convertScaleAbs(img, alpha=2.2, beta=np.random.randint(0,35))
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

        # wt, ht = self.image_size
        # h, w = img.shape
        # f = min(wt / w, ht / h)
        # tx = (wt - w * f) / 2
        # ty = (ht - h * f) / 2

        # # map image into target image
        # M = np.float32([[f, 0, tx], [0, f, ty]])
        # target = np.ones([ht, wt]) * 255
        # img = cv2.warpAffine(img, M, dsize=(wt, ht), dst=target, borderMode=cv2.BORDER_TRANSPARENT)
         
        target_width, target_height = self.image_size
        height, width = img.shape[:2]
        ratio = min(target_width / width, target_height / height)
        new_w, new_h = int(width * ratio), int(height * ratio)

        resized_image = cv2.resize(img, (new_w, new_h))
        delta_w = target_width - new_w
        delta_h =  target_height - new_h
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        padding_color = 255 # (0,0,0) if we are using rgb images

        img = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)


        # img = torch.from_numpy(img)
        # print(f'before resizing img.size(): {img.size()}')
        # img = torchvision.transforms.functional.resize(img, (32,356))
        # print('torchvision transform is working')
        #  img = img.numpy()
        
        return img, text

    @staticmethod
    def label_indexer(vocab: str, label: np.ndarray):
        """Convert label to index by vocab
        """
        # def __call__(self, data: np.ndarray, label: np.ndarray):
        return np.array([vocab.index(l) for l in label if l in vocab])

    @staticmethod
    def label_padding(padding_value: int, max_len: int, label: np.ndarray):
        label = label[:max_len]
        # put 0 padding values on the left, max_len - len(labels) padding values on the right
        return np.pad(label, (0,max_len - len(label)), mode='constant', constant_values=padding_value)
    
    #Preprocessing of single image
    def single_image_preprocessing(self, img):
        target_width, target_height = self.image_size
        height, width = img.shape[:2]
        ratio = min(target_width / width, target_height / height)
        new_w, new_h = int(width * ratio), int(height * ratio)
        resized_image = cv2.resize(img, (new_w, new_h))
        delta_w = target_width - new_w
        delta_h =  target_height - new_h
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        padding_color = 255
        img = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
        img = torch.from_numpy(img).float() 
        img = img / 255.0
        return img



