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

image = xTrain[0, :].reshape((28, 28))
plt.imshow(image)
plt.show()