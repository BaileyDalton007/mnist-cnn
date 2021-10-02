import os

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

modelList = ["1_Layer", "2_Layer", "3_Layer"]
checkpointPath = r'logs\model_saves\{}\cp'.format(modelList[0])

def create_model():
  model = tf.keras.models.Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=2), # shrinks input by a factor of two
    Dropout(0.2), # Randomly drops out connections

    Flatten(),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax') # softmax used for output layer of clustering systems
  ])

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

testIndex = 100

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