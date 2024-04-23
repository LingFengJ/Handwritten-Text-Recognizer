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
# from torch.utils.tensorboard import SummaryWriter
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
    train_dataset = HandwritingDataset(train_data, vocab, max_len)
    val_dataset = HandwritingDataset(val_data, vocab, max_len)
    test_dataset = HandwritingDataset(test_data, vocab, max_len)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print("len of loaders:",train_loader.__len__(), val_loader.__len__(), test_loader.__len__())

    def testing_images():
    #--------------------------testing--------------------------------
        import matplotlib.pyplot as plt
        
        print("len of loaders:",train_loader.__len__(), val_loader.__len__(), test_loader.__len__())
        # Get one batch of images from the train_loader
        images, temp_labels = next(iter(train_loader))

        # # Plot the first image from the batch
        # plt.imshow(images[0].squeeze(), cmap='gray')
        # plt.show()

        # Plot the first 5 images from the batch
        fig, axs = plt.subplots(1, 5, figsize=(15, 10))
        for i in range(5):
            # print(images[i].shape)
            axs[i].imshow(images[i].squeeze(), cmap='gray') #squeeze to remove channel dimension
            # axs[i].imshow(images[i].permute(1, 2, 0), cmap='gray') # if the image is rgb
            print(images[i].shape, temp_labels[i])
        plt.show()
        #--------------------------end testing--------------------------------
    testing_images()  # images are now tensors with torch.Size([32, 256])

    # def github_dataprovider():
    #     # Code from github: uses mltu library 
    #     # # Create a data provider for the dataset
    #     # data_provider = DataProvider(
    #     #     dataset=dataset,
    #     #     skip_validation=True,
    #     #     batch_size=configs.batch_size,
    #     #     data_preprocessors=[ImageReader()],
    #     #     transformers=[
    #     #         ImageResizer(configs.width, configs.height, keep_aspect_ratio=True),
    #     #         LabelIndexer(configs.vocab),
    #     #         LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
    #     #         ],
    #         # )

    #     # Split the dataset into training and validation sets
    #     # train_data_provider, val_data_provider = data_provider.split(split = 0.9)
    #     pass
    
    # # using model1
    # # input_dimension = (32,256,1) # Height, Width, Channels
    # input_dimension = 1
    # output_dimension = len(vocab) 
    # # learning_rate = 0.0005
    # learning_rate = 0.002
    # model = Model1(input_dimension, output_dimension, activation="leaky_relu", dropout=0.2)

    # blank = len(vocab)
    # criterion = CTCLoss(blank=len(vocab), reduction='mean')
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # train_epochs = 1
    # model_path = os.path.join("Results", datetime.strftime(datetime.now(), "%Y%m%d%H%M"))
    # model = model.to(device)

    # # # Define metrics
    # # cer_metric = CERMetric(vocabulary=vocab)
    # # wer_metric = WERMetric(vocabulary=vocab)

    # # # Define callbacks
    # # earlystopper = EarlyStopping(monitor="val_CER", patience=20, verbose=1, mode="min") # stop training if the metric has stopped improving
    # ## checkpoint saves the model after every epoch
    # #checkpoint = ModelCheckpoint(f"{model_path}/model.pt", monitor="val_CER", verbose=1, save_best_only=True, mode="min")
    # checkpoint = ModelCheckpoint(model, f"{model_path}/model.pt")
    # # trainLogger = TrainLogger(model_path) # logs training statistics
    # tb_callback = SummaryWriter(f"{model_path}/logs")
    # # # reduce learning rate on plateau when a metric has stopped improving
    # # reduceLROnPlat = ReduceLROnPlateau(monitor="val_CER", factor=0.9, min_delta=1e-10, patience=5, verbose=1, mode="auto")
    # # # convert model to onnx format, a platform-independent format for models
    # # model2onnx = Model2onnx(f"{configs.model_path}/model.h5")

    # # Train the model
    # # for epoch in range(train_epochs):
    # for epoch in tqdm(range(train_epochs)):
    #     model.train(True) # Set the model to training mode
    #     # for i, (inputs, targets) in enumerate(train_loader):
    #     for inputs, targets in tqdm(train_loader):
    #         inputs = inputs.unsqueeze(1)
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         # print(inputs.shape)
    #         inputs, targets = Variable(inputs), Variable(targets)
    #         optimizer.zero_grad()
    #         outputs = model(inputs)

    #         # Compute the lengths of the input and target sequences
    #         # input_lengths = torch.full(size=(inputs.size(0),), fill_value=outputs.size(1), dtype=torch.long)
    #         # target_lengths = torch.full(size=(targets.size(0),), fill_value=targets.size(1), dtype=torch.long)

    #         # Remove padding and blank tokens from target
    #         target_lengths = torch.sum(targets != blank, dim=1)
    #         using_dtype = torch.int32 if max(target_lengths) <= 256 else torch.int64
    #         device = outputs.device

    #         target_unpadded = targets[targets != blank].view(-1).to(using_dtype)

    #         outputs = outputs.permute(1, 0, 2)  # (sequence_length, batch_size, num_classes)
    #         output_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=using_dtype).to(device)

    #         loss = criterion(outputs, target_unpadded, output_lengths, target_lengths.to(using_dtype))

    #         # Compute the CTC loss
    #         #loss = torch.nn.functional.ctc_loss(outputs.log_softmax(2), targets, input_lengths, target_lengths)
            
    #         # loss = criterion(outputs, targets, input_lengths, target_lengths)
    #         loss.backward()
    #         optimizer.step()

    #         # Log metrics
    #         # cer_metric.update(outputs, targets)
    #         # wer_metric.update(outputs, targets)
    #         tb_callback.add_scalar('Loss/train', loss.item(), epoch)
    #         # tb_callback.add_scalar('CER/train', cer_metric.result(), epoch)
    #         # tb_callback.add_scalar('WER/train', wer_metric.result(), epoch)

    #     # Validation
    #     model.eval()
    #     with torch.no_grad():
    #         for i, (inputs, targets) in enumerate(val_loader):
    #             inputs = inputs.unsqueeze(1) 
    #             inputs, targets = inputs.to(device), targets.to(device)
    #             inputs, targets = Variable(inputs), Variable(targets)
    #             outputs = model(inputs)
    #             target_lengths = torch.sum(targets != blank, dim=1)
    #             using_dtype = torch.int32 if max(target_lengths) <= 256 else torch.int64
    #             device = outputs.device
    #             target_unpadded = targets[targets != blank].view(-1).to(using_dtype)
    #             outputs = outputs.permute(1, 0, 2)  # (sequence_length, batch_size, num_classes)
    #             output_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=using_dtype).to(device)
    #             loss = criterion(outputs, target_unpadded, output_lengths, target_lengths.to(using_dtype))
    #             # loss = criterion(outputs, targets)

    #             # Log metrics
    #             # cer_metric.update(outputs, targets)
    #             # wer_metric.update(outputs, targets)
    #             tb_callback.add_scalar('Loss/val', loss.item(), epoch)
    #             # tb_callback.add_scalar('CER/val', cer_metric.result(), epoch)
    #             # tb_callback.add_scalar('WER/val', wer_metric.result(), epoch)

                
    #     # Call callbacks (for each epoch)
    #     # earlystopper.step(cer_metric.result())
    #     # checkpoint.step(cer_metric.result())
    #     # trainLogger.step(cer_metric.result())
    #     # reduceLROnPlat.step(cer_metric.result())
    #     # model2onnx.step(cer_metric.result())

    #     # Reset metrics
    #     # cer_metric.reset()
    #     # wer_metric.reset()
        
    #     # Save the model
    #     # torch.save(model.state_dict(), os.path.join(model_path, 'model.pt'))
        

    # # # Save training and validation datasets as csv files
    # # train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
    # # val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))

    # torch.save(model.state_dict(), os.path.join(model_path, 'model.pt'))
    # tb_callback.close()




if __name__ == "__main__":
    main()