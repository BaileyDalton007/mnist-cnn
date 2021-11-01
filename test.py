import os

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modelDefs import cnn_1Layer, cnn_2Layer, cnn_3Layer, cnn_brownlee, cnn_brownlee_D25, cnn_brownlee_D50, cnn_3brownlee, cnn_4Layer

modelIndex = 7 # Choose what model will be evaluated

modelList = ["1_Layer", "2_Layer", "3_Layer", "brownlee", "brownlee_D25", "brownlee_D50", "3_brownlee", "4_Layer"]
cnn_models = [cnn_1Layer, cnn_2Layer, cnn_3Layer, cnn_brownlee, cnn_brownlee_D25, cnn_brownlee_D50, cnn_3brownlee, cnn_4Layer]

checkpointPath = r'logs\model_saves\{}\cp'.format(modelList[modelIndex])

def create_model():
  model = cnn_models[modelIndex]

  model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.metrics.SparseCategoricalAccuracy()])

  return model


# Creates and loads model from previous training
model = create_model()
model.load_weights(checkpointPath)

# Sets up and Reshapes data to test with
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

test_df = pd.read_csv(r'data\fashion-mnist_test.csv')
test_data = np.array(test_df, dtype='float32')

xTest = test_data[:, 1:] / 255
yTest = test_data[:, 0]
xTest = xTest.reshape(xTest.shape[0], * (28, 28, 1))


def singlePrediction(testIndex):
    # Gives image classification prediction
    img = xTest[testIndex]
    img = (np.expand_dims(img,0))
    predictions_single = model.predict(img)
    classPrediction = class_names[int(np.argmax(predictions_single[0]))]
    print(r'Prediction: {}'.format(classPrediction))

    # Displays Image and ground truth
    plt.imshow(xTest[testIndex], cmap='gray')
    print(r'Ground Truth: {}'.format(class_names[int(yTest[testIndex])]))
    plt.show()

def listPrediction(size):
    imgList = xTest[:size]
    mislabeled = []
    for i in range(len(imgList)):
        img = imgList[i]
        img = (np.expand_dims(img,0))
        predictions_single = model.predict(img)
        prediction = np.argmax(predictions_single[0])
        truth = yTest[i]
        if prediction != truth:
            mislabeled.append(i)
    return mislabeled

print(len(listPrediction(1000)))

