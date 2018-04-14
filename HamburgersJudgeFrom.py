import os

import keras
import numpy as np
from keras.layers import Activation, Dense, MaxPooling2D, Conv2D, Flatten
from keras.preprocessing.image import load_img
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
import wx
import PyQt5


FILE_PATH_TEST= "F:\Images\Hamburgers\Test"
APPLE_LABEL = 0
BANANA_LABEL = 1

trainImgs = []
trainLabels = []
testImgs = []
testLabels = []
fileNames= []

for testFold in os.listdir(FILE_PATH_TEST):
    currentDir = FILE_PATH_TEST + "/" + testFold
    label = 0
    if testFold == "Mac":
        label = APPLE_LABEL
    elif testFold == "Mos":
        label = BANANA_LABEL

    for testImg in os.listdir(currentDir):
        filePath = currentDir + "/" + testImg
        fileNames.append(testImg)
        img = np.array(load_img(filePath, target_size=(64, 64)))
        img = img.transpose([2, 0, 1])
        img = img.reshape((64, 64, 3))
        img = img / 255
        testImgs.append(img)
        testLabels.append(label)
trainImgs = np.array(trainImgs)
testImgs = np.array(testImgs)
testLabelArray = to_categorical(testLabels, 2)
print(trainImgs.ndim)

json_string = open("HamburgersModel.json", "r").read()


learnedModel = keras.models.model_from_json(json_string)
learnedModel.load_weights("processedParamHamburgers1513840869.h5")
learnedModel.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
predicts = learnedModel.predict_classes(testImgs)
print(predicts)
print(fileNames)
print(learnedModel.evaluate(testImgs, testLabelArray))
#learnedScore = learnedModel.evaluate(testImgs, testLabelArray, verbose=1)
#print(learnedScore)




