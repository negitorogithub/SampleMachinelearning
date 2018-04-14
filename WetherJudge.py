import glob
import numpy as np
import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Activation, Dense, Dropout, MaxPooling2D, Conv1D, ZeroPadding2D, Flatten
from keras.optimizers import Adagrad
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import os
FILE_PATH_TRAIN_SUMMER = "F:\Images\Seasons\Train\Summer"
FILE_PATH_TRAIN_WINTER = "F:\Images\Seasons\Train\Winter"
SUMMER_LABEL = 0
WINTER_LABEL = 1


trainImgs = []
trainLabels = []
testImgs = []
testLabels = []


for trainFold in os.listdir("F:/Images/Seasons/Train"):
    currentDir = "F:/Images/Seasons/Train"+"/"+trainFold
    label = 0
    if trainFold == "Summer":
        label = SUMMER_LABEL
    elif trainFold == "Winter":
        label = WINTER_LABEL

    for trainImg in os.listdir(currentDir):
        filePath = currentDir + "/" + trainImg
        img = np.array(load_img(filePath, target_size=(32, 32)))
        img = img.transpose([2, 0, 1])
        img = img.reshape(1, img.shape[0] * img.shape[1] * img.shape[2]).astype("float32")[0]
        img = img / 255
        trainImgs.append(img)
        trainLabels.append(label)


for testFold in os.listdir("F:/Images/Seasons/Test"):
    currentDir = "F:/Images/Seasons/Test"+"/"+testFold
    label = 0
    if testFold == "Summer":
        label = SUMMER_LABEL
    elif testFold == "Winter":
        label = WINTER_LABEL

    for testImg in os.listdir(currentDir):
        filePath = currentDir + "/" + testImg
        img = np.array(load_img(filePath, target_size=(32, 32)))
        img = img.transpose([2, 0, 1])
        img = img.reshape(1, img.shape[0] * img.shape[1] * img.shape[2]).astype("float32")[0]
        img = img / 255
        testImgs.append(img)
        testLabels.append(label)
trainImgs = np.array(trainImgs)
testImgs = np.array(testImgs)
allLabelArray = to_categorical(trainLabels, 2)
testLabelArray = to_categorical(testLabels, 2)

print(trainImgs.ndim)

model = keras.models.Sequential()
model.add(Dense(40, input_dim=3072))
model.add(Activation('softmax'))
model.add(Dense(2))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


model.fit(trainImgs, allLabelArray, batch_size=10, epochs=1000)
score = model.evaluate(testImgs, testLabelArray, verbose=1)
print(score)
print("finish")

