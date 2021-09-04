# Tensorboard server start
# tensorboard --logdir logs/cnn_1layer/

from numpy.lib.histograms import histogram
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Downloaded data set from kaggle, which is already
# pre-processed to csv form where each image is 1 row
# populated with pixel values (28*28)

train_df = pd.read_csv(r'data\fashion-mnist_train.csv')
test_df = pd.read_csv(r'data\fashion-mnist_test.csv')

# Tensorflow wants int32 or float32
train_data = np.array(train_df, dtype='float32')
test_data = np.array(test_df, dtype='float32')

# X is pixel data and Y is labels
# Pixel data is divided by 255 so that the value is between 0 and 1
# instead of between 0 and 255

xTrain = train_data[:, 1:] / 255 # excludes 0 column which is label
yTrain = train_data[:, 0] # includes 0 column which is label

xTest = test_data[:, 1:] / 255
yTest = test_data[:, 0]

# Split into training data and validation data
# Test size of 0.2 = 20% of data will be validation

xTrain, xValidate, yTrain, yValidate = train_test_split(
    xTrain, yTrain, test_size = 0.2, random_state = 12345
)

imgRows = 28
imgCols = 28
batch_size = 512
imgShape = (imgRows, imgCols, 1)

xTrain = xTrain.reshape(xTrain.shape[0], *imgShape)
xTest = xTest.reshape(xTest.shape[0], *imgShape)
xValidate = xValidate.reshape(xValidate.shape[0], *imgShape)

# Pass in a list of layers to the model
cnn_model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=imgShape),
    MaxPooling2D(pool_size=2), # shrinks input by a factor of two
    Dropout(0.2), # Randomly drops out connections

    Flatten(),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax') # softmax used for output layer of clustering systems
])

tensorboard = TensorBoard(
    log_dir=r'logs\{}'.format('cnn_1layer'),
    write_graph = True,
    write_grads=True,
    histogram_freq = 1,
    write_images = True
)

cnn_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics = ['accuracy']
)

cnn_model.fit(
    xTrain, yTrain, batch_size = batch_size,
    epochs = 10, verbose = 1,
    validation_data=(xValidate, yValidate),
    callbacks = [tensorboard]
)

# Evaluate Model
score = cnn_model.evaluate(xTest, yTest, verbose = 0)

print('test loss: {:.4f}'.format(score[0]))
print('test acc: {:.4f}'.format(score[1]))