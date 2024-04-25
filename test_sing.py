import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from os import listdir, path
from os.path import isfile, join
from preprocessor import Preprocessor

# Set the path to your folder containing images
preprocessor = Preprocessor((512, 64))

# Set the path to your folder containing images
mypath = "/Users/user/Desktop/ai_proj/Handwritten-Text-Recognizer/data/iam_sentences/dataset/"

# Get all image files in the folder
image_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# Limit to the first 5 images
image_files = image_files[:5]

# Create a plot with a single row and five columns
rows = 1
columns = len(image_files)

# Initialize the plot with specified rows and columns
fig, axes = plt.subplots(rows, columns, figsize=(20, 4))  # Adjusted figsize for a wider plot

# Ensure axes is a list or numpy array
if not isinstance(axes, (list, np.ndarray)):
    axes = [axes]

# Iterate over the image files and apply preprocessing before display
for ax, image_file in zip(axes, image_files):
    # Construct the full path to the image file
    image_path = join(mypath, image_file)

    # Read the image
    img = plt.imread(image_path)

    # Apply preprocessing to resize the image
    img = preprocessor.single_image_preprocessing(img)

    # Display the preprocessed image in the subplot
    ax.imshow(img)
    ax.axis('on')  # Hide axis ticks and labels

# Show the plot with the preprocessed images
# plt.tight_layout()  # Ensure plots don't overlap
plt.show()