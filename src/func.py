import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cv2
import numpy as np
import imutils


def load_my_image (figname):
    '''
    It takes the name of the file and loads it in grey scale
    Input: name of the file
    Output: the image
    '''
    image = cv2.imread(figname)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def crop_and_reshape (image):
    '''
    It takes an image and reshapes it in 28x28 pixels. If the original image is not a square, it crops the margins centering the image, in order to deform the original figure
    It's supposed that the original figure has a white background, it creates the negative
    Input: an image
    Output: an image in 28x28 pixels with black background
    '''
    size = image.shape
    h = size[0]
    w = size[1]
    if (h == 28) and (w == 28):
        return cv2.bitwise_not(image)
    if h > w:
        reduced = imutils.resize(image, width=28)
        mid = reduced.shape[0] // 2
        croped = reduced[(mid-14):(mid+14), 0:28]
    elif h < w:
        reduced = imutils.resize(image, height=28)
        mid = reduced.shape[1] // 2
        croped = reduced[0:28, (mid-14):(mid+14)]
    else:
        croped = imutils.resize(image, height=28)
    return cv2.bitwise_not(croped)

def fig_to_model_format (image):
    '''
    Prepares the chosen number image to be used as imput in the model
    Input: an image
    Output: an array with (1, 28, 28, 1) shape
    '''
    num_mod = image[:,:,1].reshape((1, 28, 28, 1))
    num_mod = num_mod.astype('float32') / 255
    return num_mod
