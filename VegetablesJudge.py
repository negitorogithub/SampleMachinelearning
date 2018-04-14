import os

import time
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.layers import Activation, Dense, MaxPooling2D, Conv2D, Flatten, Dropout
from keras.preprocessing.image import load_img
from keras.utils.np_utils import to_categorical
import keras.models

GENRE = "Vegetables"
JUDGE1 = "GreenPepper"
JUDGE2 = "Paprika"

FILE_PATH_TRAIN = "F:/Images/"+GENRE + "/Train"
FILE_PATH_TEST  = "F:/Images/"+GENRE + "/Test"
FILE_PATH_VALIDATION = "F:/Images/"+GENRE + "/Validation"
JUDGE1_LABEL = 0
JUDGE2_LABEL = 1

trainImgs = []
trainLabels = []
testImgs = []
testLabels = []

useOptimizer = "relu"

dataGen = ImageDataGenerator(
    featurewise_center=True,
    samplewise_center=False,
    featurewise_std_normalization=True,
    samplewise_std_normalization=True,
    zca_whitening=False,
    zca_epsilon=1e-6,
    rotation_range=180.,
    width_shift_range=0.5,
    height_shift_range=0.5,
    shear_range=0.,
    zoom_range=0.5,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=None,
)

dataGenSimple = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-6,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
)


def save_model(model2save, json_name: str):
    json = model2save.to_json()
    open(json_name, 'w').write(json)


def save_weights(model2save, h5_name: str):
    model2save.save_weights(h5_name)


for trainFold in os.listdir(FILE_PATH_TRAIN):
    currentDir = FILE_PATH_TRAIN + "/" + trainFold
    label = 0
    if trainFold == JUDGE1:
        label = JUDGE1_LABEL
    elif trainFold == JUDGE2:
        label = JUDGE2_LABEL

    for trainImg in os.listdir(currentDir):
        filePath = currentDir + "/" + trainImg
        img = np.array(load_img(filePath, target_size=(64, 64)))
        img = img.transpose([2, 0, 1])
        img = img.reshape((64, 64, 3))
        img = img / 255
        trainImgs.append(img)
        trainLabels.append(label)

for testFold in os.listdir(FILE_PATH_TEST):
    currentDir = FILE_PATH_TEST + "/" + testFold
    print(currentDir)
    label = 0
    if testFold == JUDGE1:
        label = JUDGE1_LABEL
    elif testFold == JUDGE2:
        label = JUDGE2_LABEL

    for testImg in os.listdir(currentDir):
        filePath = currentDir + "/" + testImg
        print(filePath)
        img = np.array(load_img(filePath, target_size=(64, 64)))
        img = img.transpose([2, 0, 1])
        img = img.reshape((64, 64, 3))
        img = img / 255
        testImgs.append(img)
        testLabels.append(label)
trainImgs = np.array(trainImgs)
testImgs = np.array(testImgs)
allLabelArray = to_categorical(trainLabels, 2)
testLabelArray = to_categorical(testLabels, 2)
print(trainImgs.ndim)
print(testImgs.ndim)
open(GENRE+"Model.json", 'a')
json_string = open(GENRE+"Model.json", "r").read()
"""
learnedModel = keras.models.model_from_json(json_string)
learnedModel.load_weights("param.h5")
learnedModel.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
learnedScore = learnedModel.evaluate(testImgs, testLabelArray, verbose=1)
print(learnedScore)
learnedModel.fit(trainImgs, allLabelArray, batch_size=64, epochs=100,validation_split=0.2,
          callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)]
         )
learnedScore = learnedModel.evaluate(testImgs, testLabelArray, verbose=1)
print(learnedScore)

"""


while True:
    model = keras.models.Sequential()
    model.add(Dense(60, input_shape=(64, 64, 3)))
    model.add(Activation(useOptimizer))
    model.add(Conv2D(5, 2))
    model.add(Activation(useOptimizer))

    model.add(Conv2D(5, 2))
    model.add(Activation(useOptimizer))
    model.add(MaxPooling2D((2, 2), 2, padding='valid', data_format="channels_last"))
    model.add(Dropout(0.1))
    model.add(Conv2D(5, 2))
    model.add(Activation(useOptimizer))
    model.add(Conv2D(5, 2))
    model.add(Activation(useOptimizer))
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(2))
    model.add(Activation("softmax"))

    save_model(model, GENRE+"Model.json")
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    """
    model.fit(trainImgs, allLabelArray, batch_size=64, epochs=200,validation_split=0.2,
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=20),
              keras.callbacks.ModelCheckpoint(filepath="trueParams.h5",monitor="val_acc",save_best_only=True)]
              )

    """

    model.fit_generator(
                    dataGenSimple.flow(trainImgs, allLabelArray, batch_size=64),
                    epochs=2000,
                    callbacks=[keras.callbacks.EarlyStopping(monitor="loss", patience=10),
                               keras.callbacks.ModelCheckpoint(
                                   filepath="processedParam"+GENRE+str(int(time.time()))+".h5",
                                   monitor="loss",
                                   save_best_only=True)
                               ],
                    verbose=1,
                    steps_per_epoch=1
    )
    score = model.evaluate(testImgs, testLabelArray, verbose=1)
    print(score)
