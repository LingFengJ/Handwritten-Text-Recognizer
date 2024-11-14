# Handwritten-Text-Recognizer
Handwritten text Recognizer is an OCR application that aims to recognize handwritten english sentences written in cursive. 
It is implemented in Pytorch through fine-tuned Resnet archictures and LSTM for sequence processing and trained using CTC loss.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Visualisation](#visualisation)
- [Features](#features)
- [Authors](#authors)
- [License](#license)


## Installation
**Clone the repo:**
  ```bash
  git clone https://github.com/LingFengJ/Handwritten-Text-Recognizer
```
<br>
    
**Install Python dependencies**
 ```bash
  pip install -r Handwritten-Text-Recognizer/requirements.txt
```
## Usage
### Dataset
  After installation, download the ***sentence*** dataset from
   <a href="https://fki.tic.heia-fr.ch/databases/iam-handwriting-database">IAM Handwriting Database</a>
   <br>
   *Or*
   <br>
   From the following
   <a href="https://www.kaggle.com/datasets/debadityashome/iamsentences/code"> Kaggle Link </a>
   <br>
   <br>
   unzip in the **data** folder with the name <strong>iam_sentences</strong>
   <br>
### How to run the project:
  - You can download our <a href="https://drive.google.com/drive/folders/14EmkyGDwhCRimP2kF7oa1mjRwSXgk2uV"> Pretrained Model </a>
  - Train your own model:
``` bash
cd path/to/Handwritten-Text-Recognizer
python train.py
```
## Visualisation 

### Testing:
```
python testing_.py
tensorboard --logdir=path/to/Handwritten-Text-Recognizer/Results
```
The ***Character Error Rate***(CER) and ***Word Error Rate***(WER) will be visible on <a href="http://localhost:6006/">Tensorboard </a>

## Features
A [short presentation](https://www.canva.com/design/DAGHiM322vs/zk8GCeYG0vYzkjkzjOjKWA/watch?utm_content=DAGHiM322vs&utm_campaign=designshare&utm_medium=link&utm_source=editor) about the project
<br>
![image](https://github.com/user-attachments/assets/795d2061-3b1c-44aa-85fe-60e8c60a960a)




## Authors
  We are a group of bachelor students of Applied Computer Science and Artificial Intelligence at the Sapienza University of Rome. 
  The Handwritten-Text-Recognizer is a project that belongs to our academic curriculum - it is designed for fullfill exam requirements of the examination "AI Lab: Computer Vision and NLP". <br>
  The process is stimulating and we benefited a lot by tackling challenges in sequential learning and deep neural network architecture design. For any clarification or further information regarding the project, please fell free to reach out to us.
- Lingfeng Jin
- Abduazizkhon Shomansurov
- Liyu Jin
- Gioia Zheng

## License
Handwritten-Text-Recogniser is released under [MIT License](./LICENSE)



