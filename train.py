from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import os
import torch
import numpy as np
from datetime import datetime
from itertools import groupby

import cv2
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Grayscale
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import HandwritingDataset


# import various models to test
from model1 import Model as Model1
# from modelres import Model as ModelRes # model with pretrained resnet on torchvision.models
# from model2 import Network as Model2
# from crnn import CRNN

import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.nn import CTCLoss

from utils import EarlyStopping, ModelCheckpoint
from utils import label_to_strings
from torchmetrics.text import CharErrorRate 
from torchmetrics.text import WordErrorRate
from torch.optim.lr_scheduler import ReduceLROnPlateau

def set_device():
  """
  Set the device. CUDA if available, CPU otherwise
  """
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device != "cuda":
    print("GPU is not enabled in this notebook(if you are training with colab). \n"
          "If you want to enable it, in the menu under `Runtime` -> \n"
          "`Hardware accelerator.` and select `GPU` from the dropdown menu")
  else:
    print("GPU is enabled in this device/notebook. \n"
          "If you want to disable it, in the menu under `Runtime` -> \n"
          "`Hardware accelerator.` and select `None` from the dropdown menu")

  return device

def decode(vocab, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.LongTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.LongTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        blank = len(vocab)
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([vocab[i] for i in t if i<blank])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != blank and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(vocab[t[i]])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    decode(
                        t[index:index + l], torch.LongTensor([l]), raw=raw))
                index += l
            return texts

def label_to_string(predictions: np.ndarray, vocabulary):
        """Convert predictions to strings
        """
        # use argmax to find the index of the highest probability
        argmax_preds = np.argmax(predictions, axis=-1)
        
        # use groupby to find continuous same indexes,we could use torch.unique_consecutive as well
        grouped_preds = [[k for k,_ in groupby(preds)] for preds in argmax_preds]

        texts = ["".join([vocabulary[k] for k in group if k < len(vocabulary)]) for group in grouped_preds]

        return texts

def train_model(train_loader, val_loader, test_loader, vocab, max_len, device = 'cpu', train_epochs=400, trained = None, trained_path = ""):
    input_dimension = 1
    # input_dimension = (1, 32, 256) # channels first for conv2d
    output_dimension = len(vocab) 
    # learning_rate = 0.0005 
    # learning_rate = 0.002
    learning_rate = 0.0012 
    
    last_train_path = os.path.join('Model', 'model.pt')
    if os.path.exists(last_train_path):
        print('loading the last trained model')
        model = Model1(input_dimension, output_dimension, activation="leaky_relu", dropout=0.2)
        state_dict = torch.load(last_train_path, map_location=torch.device(device))
        model.load_state_dict(state_dict)
        model = model.to(device)
        # learning_rate = 0.0004
        learning_rate = 9e-05
    else: 
        #crnn = net_init()
        # print(crnn)
        #model = crnn
        # model = ModelRes(input_dimension, output_dimension, activation="leaky_relu", dropout=0.2)
        #model = Model(output_channels=len(vocab), activation="leaky_relu", dropout=0.2)
        model = Model1(input_dimension, output_dimension, activation="leaky_relu", dropout=0.2)
        # model = Model2(num_chars=len(vocab), activation="leaky_relu", dropout=0.2)
        # model = Model3(input_dim=input_dimension, output_dim=output_dimension, activation="leaky_relu", dropout=0.2)
        learning_rate = 0.0005

    blank = len(vocab)
    print(f'blank token: {blank}')
    criterion = CTCLoss(blank=len(vocab), reduction='mean', zero_infinity=True)
    # criterion = criterion.to(device)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    model_path = os.path.join("Results", datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M"))
    model = model.to(device)

    # # Define metrics
    cer_metric = CharErrorRate()
    wer_metric = WordErrorRate()

    # # Callbacks
    earlystopper = EarlyStopping(patience=20)
    checkpoint = ModelCheckpoint(model, f"{model_path}/model.pt")
    tensorboard = SummaryWriter(f"{model_path}/logs")
    LR_reducer = ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-10, patience=5, mode="min")

    # Train the model
    # per_epoch = 100 # print one prediction per 50 iteration to see the progress
    per_epoch = True # print the predictions only in first k epochs 
    # tempfile = open('tempfile.txt', 'w')
    for epoch in tqdm(range(train_epochs)):
        per_epoch = True
        trial = False
        model.train(True) # Set the model to training mode
        for i, (inputs, targets) in tqdm(enumerate(train_loader)):
        # for inputs, targets in tqdm(train_loader):
            inputs = inputs.unsqueeze(1)
            # inputs = inputs.permute(0,3,1,2) # (batch_size, channels, height, width)
            inputs, targets = inputs.to(device), targets.to(device)
            # print(inputs.dtype, inputs.shape)
            inputs, targets = Variable(inputs), Variable(targets)
            optimizer.zero_grad()
            outputs = model(inputs)

            # print(f'the outputs are {outputs}')
            # print(outputs.shape)
            
            # print(f'predictions(model3): {decoder(outputs)}')
            # print(f'the right answer {label_to_string(targets)}')
            # print(f'the prediction {label_to_string(outputs)}')
            # input_lengths = torch.full(size=(inputs.size(0),), fill_value=outputs.size(1), dtype=torch.long)
            # target_lengths = torch.full(size=(targets.size(0),), fill_value=targets.size(1), dtype=torch.long)

            # Remove padding and blank tokens from target
            target_lengths = torch.sum(targets != blank, dim=1) #list of lengths of each target
            device = outputs.device

            target_unpadded = targets[targets != blank].view(-1).to(torch.int32) # remove padding and flatten the tensor

            outputs_temp = outputs.permute(1, 0, 2)  # (sequence_length, batch_size, num_classes)
            output_lengths = torch.full(size=(outputs_temp.size(1),), fill_value=outputs_temp.size(0), dtype=torch.int32).to(device)
            # print(f'outputs.size(): {outputs.size()}')
            # print(f'target_unpadded.size(): {target_unpadded.size()}')
            # print(f'target size: {targets.size()}')
            # print(f'unpadded target: {target_unpadded}')
            # print(f'predictions: {label_to_string(outputs[0].detach().numpy(), vocab)}')

            if per_epoch:
                per_epoch = False
                # torch.set_printoptions(profile="full")
                # print(outputs, file=tempfile)
                # tempfile.flush()
                # print(f'output length: {output_lengths}')
                # print(f'target length: {target_lengths}')
                preds, answers = label_to_strings(outputs.detach().cpu().numpy(), targets.cpu().numpy(), vocab)
                # print(f'outpus shape after permute: {outputs.shape}')
                print(f'we are at epoch {epoch+1} and batch {i+1} of training')
                print(f'predictions len: {len(preds)}')
                print(f'predictions: {preds}')
                # print(f'answers: {answers}')
                # print(f'answers: {answers}', file=tempfile)
                print(f'our learning rate is {optimizer.param_groups[0]["lr"]}')
                if i:
                    print(f'we have cer_metric: {cer_metric.compute()} and wer_metric: {wer_metric.compute()}')

            loss = criterion(outputs_temp, target_unpadded, output_lengths, target_lengths.to(torch.int32))
            # print(f'loss : {loss}')
            # print(isinstance(loss, torch.Tensor))

            # print(f'predictions: {label_to_string(outputs.detach().numpy(), vocab)}')
            # Compute the CTC loss
            #loss = torch.nn.functional.ctc_loss(outputs.log_softmax(2), targets, input_lengths, target_lengths)

            loss.backward()
            optimizer.step()

            if device == "cuda":
                torch.cuda.synchronize()

            # Log metrics
            preds, answers = label_to_strings(outputs.detach().cpu().numpy(), targets.detach().cpu().numpy(), vocab)
            cer_metric.update(preds, answers)
            wer_metric.update(preds, answers)

            # tb_callback.add_scalar('Loss/train', loss.item(), i) #comment this when we have gpu to run multiple epochs
            # tb_callback.add_scalar('CER/train', cer_metric.compute(), i)
            # tb_callback.add_scalar('WER/train', wer_metric.compute(), i)
            # print(f'cer_metric: {cer_metric.compute()}')

            tensorboard.add_scalar('Loss/train', loss.item(), epoch)
            tensorboard.add_scalar('CER/train', cer_metric.compute(), epoch)
            tensorboard.add_scalar('WER/train', wer_metric.compute(), epoch)
        # tempfile.close()

        # Reset metrics for each epoch
        cer_metric.reset()
        wer_metric.reset()

        # Validation
        model.eval()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.unsqueeze(1) 
                # inputs = inputs.permute(0,3,1,2)
                inputs, targets = inputs.to(device), targets.to(device)
                inputs, targets = Variable(inputs), Variable(targets)
                #print(inputs.dtype, inputs.shape)
                outputs = model(inputs)
                target_lengths = torch.sum(targets != blank, dim=1)
                using_dtype = torch.int32 if max(target_lengths) <= 256 else torch.int64
                device = outputs.device
                target_unpadded = targets[targets != blank].view(-1).to(using_dtype)
                outputs_temp = outputs.permute(1, 0, 2)  # (sequence_length, batch_size, num_classes)
                output_lengths = torch.full(size=(outputs_temp.size(1),), fill_value=outputs_temp.size(0), dtype=using_dtype).to(device)
                loss = criterion(outputs_temp, target_unpadded, output_lengths, target_lengths.to(using_dtype))
                # loss = criterion(outputs, targets)
                if not trial:
                    # print one prediction per epoch just to see the progress
                    outputs = outputs.cpu()
                    # print(f'predictions at epoch {epoch}: {label_to_string(outputs.detach().numpy(), vocab)}')
                    print(f'predictions at validation epoch {epoch}: {label_to_string(outputs.detach().numpy(), vocab)}')
                    trial = True

                if device == "cuda":
                    torch.cuda.empty_cache()
                # Log metrics                
                preds, answers = label_to_strings(outputs.detach().cpu().numpy(), targets.cpu().numpy(), vocab)
                cer_metric.update(preds, answers)
                wer_metric.update(preds, answers)
                # cer_metric.update(outputs, targets)
                # wer_metric.update(outputs, targets)
                # tb_callback.add_scalar('Loss/val', loss.item(), i) #comment this when we have gpu to run multiple epochs
                # tb_callback.add_scalar('CER/val', cer_metric.compute(), i)
                # tb_callback.add_scalar('WER/val', wer_metric.compute(), i)

                tensorboard.add_scalar('Loss/val', loss.item(), epoch)
                tensorboard.add_scalar('CER/val', cer_metric.compute(), epoch)
                tensorboard.add_scalar('WER/val', wer_metric.compute(), epoch)

                
        # Call callbacks (for each epoch)
        cer_result = cer_metric.compute()
        print(f'cer at epoch {epoch}: {cer_result}')
        earlystopper.step(cer_result)
        checkpoint.step(cer_result)
        LR_reducer.step(cer_result)
        if earlystopper.stop:
            print(f"Early stopping on epoch {epoch} with CER: {cer_result}")
            break
        # trainLogger.step(cer_metric.result())

        # Reset metrics
        cer_metric.reset()
        wer_metric.reset()
        

    torch.save(model.state_dict(), os.path.join(model_path, 'model.pt'))
    tensorboard.close()
    return 

def main():
    device = set_device()
    # Set the seed
    seed = 2024
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    metadata_folder = os.path.join('data', 'iam_sentences', 'metadata', 'sentences.txt')
    images_folder = os.path.join('data', 'iam_sentences', 'dataset')


    all_metadata = open(metadata_folder, "r").readlines()    
    
    max_len = 0
    vocab = set()
    dataset = []

    for line in tqdm(all_metadata):
        if line.startswith("#"):
            continue

        line_split = line.split(" ")
        if line_split[2] == "err":
            continue

        file_name = line_split[0] + ".png"
        label = line_split[-1].rstrip('\n')

        label = label.replace('|', ' ')

        check_path = os.path.join(images_folder, file_name)  
        if not os.path.exists(check_path): 
            continue

        dataset.append([check_path, label])
        vocab.update(list(label))
        max_len = max(max_len, len(label))

    vocab = ''.join(sorted(vocab)) # vocabulary: a string containing all possible characters
    path_to_save_vocab = os.path.join('vocab.txt')
    print(f'vocabolary of this training: {vocab}')
    with open(path_to_save_vocab, 'w') as f:
        f.write(vocab)


    # We will use cross validation 
    cross_validation = True
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if cross_validation:
        # split beforehand to avoid data leakage so that we are sure that the model never seen any image in the test_data
        train_val_data, test_data = train_test_split(dataset, test_size=0.1, random_state=42)

        dataset = train_val_data #doing this we actually destroyed the original dataset
        # but it is fine since we are not going to use test_data in this training session
        for i, (train_index, val_index) in enumerate(kf.split(dataset)):
            print(f"Cross validation: {i+1}")
            train_data = [dataset[i] for i in train_index]
            val_data = [dataset[i] for i in val_index]
            # test_data = [dataset[i] for i in test_index]
            # val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)
            print(f"train_data: {len(train_data)}")
            print(f"val_data: {len(val_data)}")
            # print(f"test_data: {len(test_data)}")

            train_dataset = HandwritingDataset(train_data, vocab, max_len, augmentations=True)
            val_dataset = HandwritingDataset(val_data, vocab, max_len, augmentations=False)
            test_dataset = HandwritingDataset(test_data, vocab, max_len, augmentations=False)

            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers = 4)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            print("len of loaders:",train_loader.__len__(), val_loader.__len__(), test_loader.__len__())
            # train the model
            train_model(train_loader, val_loader, test_loader, vocab, max_len, device=device, train_epochs=450)
    else:
        # Split the dataset into training, validation, and test sets
        train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
        val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

        # Create dataset objects
        train_dataset = HandwritingDataset(train_data, vocab, max_len, augmentations=True)
        val_dataset = HandwritingDataset(val_data, vocab, max_len, augmentations=False)
        test_dataset = HandwritingDataset(test_data, vocab, max_len, augmentations=False)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers = 4)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        print("len of loaders:",train_loader.__len__(), val_loader.__len__(), test_loader.__len__())
        train_model(train_loader, val_loader, test_loader, vocab, max_len, device=device, train_epochs=600)

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
    # testing_images()  # images are now tensors with torch.Size([32, 256])




if __name__ == "__main__":
    main()