# -*- coding: utf-8 -*-

import keras
import numpy as np
from keras.preprocessing.image import load_img
from kivy.app import App
from kivy.core.window import Window
from kivy.properties import ObjectProperty
from kivy.uix.gridlayout import GridLayout

ZERO_CLASS = "GreenPepper"

ONE_CLASS  = "Paprika"

json_string = open("F:/SampleMachinelearning/VegetablesModel.json", "r").read()
learnedModel = keras.models.model_from_json(json_string)
learnedModel.load_weights("F:\SampleMachinelearning\processedParamVegetables1515569217.h5")
learnedModel.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])


class TestApp(App):
    nextIndex = 0

    def build(self):
        self.root = MyAllWidgets()
        Window.bind(on_dropfile=self._on_file_drop)
        pass

    def _on_file_drop(self, window, file_path: bytes):
        print(type(file_path))
        if TestApp.nextIndex == 0:
            image2Indicate = self.root.judgeImage1
            label2Indicate = self.root.judgeLabel1
        elif TestApp.nextIndex == 1:
            image2Indicate = self.root.judgeImage2
            label2Indicate = self.root.judgeLabel2
        elif TestApp.nextIndex == 2:
            image2Indicate = self.root.judgeImage3
            label2Indicate = self.root.judgeLabel3

        image2Indicate.source = file_path.decode('utf-8')

        predicts = learnedModel.predict_classes(filepath2NumpyImage(file_path.decode('utf-8'))[np.newaxis, :, :, :])

        predicted_class = "?"

        if predicts[0] == 0:
            predicted_class = ZERO_CLASS
        elif predicts[0] == 1:
            predicted_class = ONE_CLASS
        label2Indicate.text = predicted_class
        TestApp.nextIndex = TestApp.nextIndex + 1
        if TestApp.nextIndex > 2:
            TestApp.nextIndex = 0
        return


class MyAllWidgets(GridLayout):
    judgeImage1 = ObjectProperty(None)
    judgeImage2 = ObjectProperty(None)
    judgeImage3 = ObjectProperty(None)
    judgeLabel1 = ObjectProperty(None)
    judgeLabel2 = ObjectProperty(None)
    judgeLabel3 = ObjectProperty(None)

    pass


def filepath2NumpyImage(filePath: str):
    img = np.array(load_img(filePath, target_size=(64, 64)))
    img = img.transpose([2, 0, 1])
    img = img.reshape((64, 64, 3))
    img = img / 255
    return img

TestApp().run()
