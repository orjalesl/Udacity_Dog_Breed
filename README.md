# Udacity_Dog_Breed
 
## Table of contents

- [Background](#background)
- [Installation](#installation)
- [Intstructions](#instructions)
- [Consideration](#consideration)


## Background

Convolutional Neural Networks (CNN's) are a type of deep learning algorithm that commonly used for computer vision. They are designed to be able to automatically and adaptively learn features from images. This project was motivated to create a tool to:
1. Detect whether an image contained a human, dog, or none.
2. If the image was a dog, then predict the dog's breed
3. If the image was a human, output the dog breed the model predicted based off the human's picture

## Installation
  The libraries used in this project are all native to Anaconda enviroment. The following libraries were used for the project:
```bash
# libraries for data and visualization
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import ImageFile
import random

# libraries for reading in image files
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

# libraries for CNN modeling
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image                  
from tqdm import tqdm
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint  
from extract_bottleneck_features import *

# library for scoring
from sklearn.metrics import classification_report
```

## Instructions

1. Download the data with the following link:
- Dog Data Set: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
- Human Data Set: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip
- VGG-16 bottleneck features: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz

2. Save the files in the same repository as the python file

3. Download the necessary libraries mentioned above.

## Consideration
To further improve the algorithm, I would consider the following:
1. Get more volume of data for the dog breeds in the model and make sure they are all represented evenly
2. Use other layers in the model to see if other parameters would provide a higher precision score
3. Use other CNN algorithms such as ResNet-50, Inception, or Xception to see which one  could perform better.


