import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cv2
import numpy as np
import imutils
from keras.models import load_model
model = load_model('my_model_my_numbers.h5')



def load_my_image(figname):
    '''
    It takes the name of the file and loads it in grey scale
    Input: name of the file
    Output: the image
    '''
    image = cv2.imread(figname)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def crop_and_reshape_numbers(image):
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
        reduced = imutils.resize(image, height=29)
        croped = reduced[0:28, 0:28]
    return cv2.bitwise_not(croped)


def fig_to_model_format(image):
    '''
    Prepares the chosen number image to be used as imput in the model
    Input: an image
    Output: an array with (1, 28, 28, 1) shape
    '''
    graynum = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    num_mod = graynum.reshape((1, 28, 28, 1))
    num_mod = num_mod.astype('float32') / 255
    #num_mod = image[:,:,1].reshape((1, 28, 28, 1))
    #num_mod = num_mod.astype('float32') / 255
    return num_mod


def sudoku_cut_frame(namefile):
    sudoku = cv2.imread(namefile)
    originalside1 = sudoku.shape[0]
    originalside2 = sudoku.shape[1]
    originalarea = originalside1*originalside1
    
    imgray = cv2.cvtColor(sudoku, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area_max = 0
    cnt = contours[1]
    for cont in contours[1:]:
        area = cv2.contourArea(cont)
        if area > area_max:
            area_max = area
            cnt = cont
    if originalarea/area_max > 20:
        return sudoku
    
    frame = cv2.drawContours(sudoku, [cnt], 0, (0,255,0), 3)
    h1 = cnt[0,0,1]
    h2 = cnt[4,0,1]
    w1 = cnt[1,0,0]
    w2 = cnt[5,0,0]

    cropped = sudoku[h1:h2, w1:w2]
    return cropped 


def sudoku_split_81(figure):     
    h = figure.shape[0] // 9
    w = figure.shape[1] // 9
    subpics = []
    w_i = 1
    w_f = 1
    h_i = 1
    h_f = 1
    for i in range(9):
        h_f += h
        for j in range(9):
            w_f += w
            cropped = figure[h_i:h_f, w_i:w_f]
            #Reduce the frame of each square:
            h_num = cropped.shape[0]
            h_var = (h_num*11) // 100
            w_num = cropped.shape[1]
            w_var = (w_num*11) // 100
            reduced = cropped[h_var:(h-h_var), w_var:(w-w_var)]
            subpics.append(reduced)
            w_i += w
        h_i += h
        w_i = 1
        w_f = 1
    return subpics


def change_contrast(image):
    a = 1.5 # 1.0-3.0
    b = 0 # 0-100
    adjusted = cv2.convertScaleAbs(image, alpha=a, beta=b)
    return adjusted


def average_pixel_color(num):
    graynum = cv2.cvtColor(num, cv2.COLOR_BGR2GRAY)
    long = graynum.shape[0]*graynum.shape[1]
    graynum = graynum.reshape((long))
    average_color = sum(graynum)/long
    return average_color


def number_or_not(sudolist):
    numnot = []
    for square in sudolist:   
        contrast = change_contrast(square)
        cut = crop_and_reshape_numbers (contrast)
        scale = average_pixel_color(cut)
        if scale < 5:
            numnot.append(0)
        else:
            numnot.append(1)
    return numnot


def from_pic_to_numbers(sudolist):
    numnot = []
    probabilidades = []
    numeromodificado = []
    
    for square in sudolist: 
        
        a = 1.5 # 1.0-3.0
        b = 0 # 0-100
        contrast = cv2.convertScaleAbs(square, alpha=a, beta=b)
        
        cut = crop_and_reshape_numbers (contrast)
        scale = average_pixel_color(cut)
        
        if scale < 9:
            numnot.append(0)
            probabilidades.append(0)
            numeromodificado.append(0)

        else:           
            #kernel = np.ones((1,1), np.uint8) 
            #img_erosion = cv2.erode(cut, kernel, iterations=2) 
            #img_dilation = cv2.dilate(cut, kernel, iterations=1)
            
            ksize = (2, 2) 
            blur = cv2.blur(cut, ksize) 
            
            squareform = fig_to_model_format(blur)
            
            res = np.argmax(model.predict(squareform), axis=-1)
            
            numnot.append(res[0])
            probabilidades.append(model.predict(squareform))
            numeromodificado.append(blur)
            
    return numnot, probabilidades, numeromodificado


def get_rows(sudokulist):
    rows = [sudokulist[(i*9):(i*9+9)] for i in range(9)]
    return rows


def get_cols(sudokulist):
    columns = [sudokulist[i::9] for i in range(9)]    
    return columns
    

def get_quads(sudokulist):
    triplets = (np.array_split(sudokulist, 27))
    quadrants = []
    counter = 0
    while counter < 26:
        for i in range(3):
            cuad = np.concatenate((triplets[counter+i], triplets[counter+i+3], triplets[counter+i+6]))
            cuad = cuad.tolist()
            quadrants.append(cuad)
        counter+=9       
    return quadrants


def sudoku_proofreader(rows, cols, quads):
    OK = True
    for f in rows:
        if len(set(f)) != 9:
            OK = False
    for c in cols:
        if len(set(c)) != 9:
            OK = False
    for q in quads:
        if len(set(q)) != 9:
            OK = False
    return OK


def correct_or_not(rows, cols, quads):
    if sudoku_proofreader(rows, cols, quads):
        print("The answer is good, well done! ;)")
    else:
        print("Errors were made, good luck next time :(")
       

def plot_sudoku(list_numbers):
    l = list_numbers
    print(f"{l[0]} {l[1]} {l[2]} | {l[3]} {l[4]} {l[5]} | {l[6]} {l[7]} {l[8]}")
    print(f"{l[9]} {l[10]} {l[11]} | {l[12]} {l[13]} {l[14]} | {l[15]} {l[16]} {l[17]}")
    print(f"{l[18]} {l[19]} {l[20]} | {l[21]} {l[22]} {l[23]} | {l[24]} {l[25]} {l[26]}")
    print("-"*21)
    print(f"{l[27]} {l[28]} {l[29]} | {l[30]} {l[31]} {l[32]} | {l[33]} {l[34]} {l[35]}")
    print(f"{l[36]} {l[37]} {l[38]} | {l[39]} {l[40]} {l[41]} | {l[42]} {l[43]} {l[44]}")
    print(f"{l[45]} {l[46]} {l[47]} | {l[48]} {l[49]} {l[50]} | {l[51]} {l[52]} {l[53]}")
    print("-"*21)
    print(f"{l[54]} {l[55]} {l[56]} | {l[57]} {l[58]} {l[59]} | {l[60]} {l[61]} {l[62]}")
    print(f"{l[63]} {l[64]} {l[65]} | {l[66]} {l[67]} {l[68]} | {l[69]} {l[70]} {l[71]}")
    print(f"{l[72]} {l[73]} {l[74]} | {l[75]} {l[76]} {l[77]} | {l[78]} {l[79]} {l[80]}")