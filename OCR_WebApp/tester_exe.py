import cv2
import numpy as np
import base64

from preprocessor import Preprocessor
import torch
from model1 import Model
import os
from itertools import groupby


def label_to_string(predictions: np.ndarray, vocabulary):
        argmax_preds = np.argmax(predictions, axis=-1)
        grouped_preds = [[k for k,_ in groupby(preds)] for preds in argmax_preds]
        texts = ["".join([vocabulary[k] for k in group if k < len(vocabulary)]) for group in grouped_preds]
        return texts

"""just as tester py file to simulate the trained application, take a image base64 convert it to np"""
def convert(image_data):
    with open('vocab.txt', 'r') as f:
        vocab = f.read().rstrip()
    model_path = os.path.join('Model', 'model.pt')
    model = Model(1, len(vocab))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    # Convert base64 image data to OpenCV image
    decoded_data = base64.b64decode(image_data)
    np_data = np.fromstring(decoded_data, np.uint8)
    img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

    # Perform processing using OpenCV
    # Example: Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_preprocessor = Preprocessor(None)
    img = img_preprocessor.single_image_preprocess(gray_img)
    img = img.unsqueeze(0) # add batch dimension
    img = img.unsqueeze(1) # add channel dimension
    prediction = model(img)
    processed_text = label_to_string(prediction, vocab)


    # Perform OCR or any other processing as required
    # processed_text = "Text extracted from image: ABC123"

    return processed_text
