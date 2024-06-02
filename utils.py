# Python
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import numpy as np
from sklearn.metrics import accuracy_score
# from torchmetrics.text import CharErrorRate, WordErrorRate
from torchmetrics.functional.text import char_error_rate as cer
from torchmetrics.functional.text import word_error_rate as wer
from itertools import groupby

def label_to_strings(predictions, targets, vocab):
    # use argmax to find the index of the highest probability
    # shape of predictions: (seq_len, batch_size, num_classes)
    argmax_preds = np.argmax(predictions, axis=-1) # now the shape is (seq_len, batch_size)

    # print('shape after taking argmax: ', argmax_preds.shape)
    grouped_preds = [[k for k,_ in groupby(preds)] for preds in argmax_preds]

    # argmax_preds = torch.argmax(output, dim=-1)
    # grouped_preds = [torch.unique_consecutive(preds) for preds in argmax_preds]

    texts = ["".join([vocab[k] for k in group if k < len(vocab)]) for group in grouped_preds]
    target_texts = ["".join([vocab[k] for k in group if k < len(vocab)]) for group in targets]

    return texts, target_texts

def label_to_string2(pred, targets, vocab):
   # Convert probability output to string
    # alphabet = """_!?#&|\()[]<>*+,-.'"€$£$§=/⊥0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzéèêâàù """
    alphabet = vocab + "_"
    cdict = {c: i for i, c in enumerate(alphabet)}
    icdict = {i: c for i, c in enumerate(alphabet)}
    tdec = pred.argmax(2).permute(1, 0).cpu().numpy().squeeze()
    # print(tdec)
    # print(tdec.ndim)
    # Convert path to label, batch has size 1 here
    target_strings = ["".join([vocab[k] for k in group if k < len(vocab)]) for group in targets]
    if tdec.ndim == 0:
        dec_transcr = ''.join([icdict[tdec.item()]]).replace('_', '')
    else:
        tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
        dec_transcr = ''.join([icdict[t] for t in tt]).replace('_', '')
    return dec_transcr

# Metrics ------------------------------------------------------------------------------------------
class Metric:
    def __init__(self):
        self.reset()

    def reset(self):
        self.values = []

    def update(self, y_pred, y_true):
        raise NotImplementedError

    def result(self):
        return np.mean(self.values)

class CERMetric(Metric):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.count = 0
        self.total = 0

    def update(self, y_pred, y_true):
        # Assuming y_pred and y_true are 1D tensors
        # y_pred = y_pred.argmax(dim=1)
        # cer = 1 - accuracy_score(y_true.numpy(), y_pred.numpy())
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        y_pred, y_true = label_to_strings(y_pred, y_true, self.vocab)
        # print(f'y_pred: {y_pred} with shape: {len(y_pred)}')
        # print(f'y_true: {y_true} with shape: {len(y_true)}')
        value = cer(y_true, y_pred)
        print(f'the cer value is: {value} with value: {value.item()}')
        self.total += value.item()
        self.count += 1
        # self.values.append(value.item())
    
    def result(self):
        return self.total / self.count if self.count > 0 else 1
    
    def reset(self):
        self.values = []
        self.total = 0
        value = 0
        self.count = 0


class WERMetric(Metric):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.count = 0
        self.total = 0

    def update(self, y_pred, y_true):
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        y_pred, y_true = label_to_strings(y_pred, y_true, self.vocab)
        # print(f'y_pred: {y_pred}')
        # print(f'y_true: {y_true}')
        value = wer(preds = y_true, target = y_pred)
        # self.values.append(value.item())
        self.total += value.item()
        self.count += 1
    
    def result(self):
        return self.total / self.count if self.count > 0 else 1
    
    def reset(self):
        self.values = []
        self.total = 0
        value = 0
        self.count = 0

# Callbacks ------------------------------------------------------------------------------------------
class Callback:
    def step(self, metric_result):
        raise NotImplementedError

class EarlyStopping(Callback):
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.at_least = 0.5
        self.best_score = None
        self.stop = False

    def step(self, metric_result):
        score = metric_result
        if self.best_score is None:
            self.best_score = score
        elif score >= self.best_score:
            self.counter += 1
            # if self.counter >= self.patience and metric_result < self.at_least:
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = score
            self.counter = 0

class ModelCheckpoint(Callback):
    def __init__(self, model, filepath):
        self.model = model
        self.filepath = filepath
        self.best_score = None

    def step(self, metric_result):
        if self.best_score is None or metric_result < self.best_score:
            self.best_score = metric_result
            torch.save(self.model.state_dict(), self.filepath)


