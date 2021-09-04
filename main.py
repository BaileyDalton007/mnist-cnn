import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Downloaded data set from kaggle, which is already
# pre-processed to csv form where each image is 1 row
# populated with pixel values

train_df = pd.read_csv(r'data\fashion-mnist_train.csv')
test_df = pd.read_csv(r'data\fashion-mnist_test.csv')

print(train_df.head())

print('done')