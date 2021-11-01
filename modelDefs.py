import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization

imgShape = (28, 28, 1)

name = '1_Layer'
cnn_1Layer = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=imgShape),
    MaxPooling2D(pool_size=2), # shrinks input by a factor of two
    Dropout(0.2), # Randomly drops out connections

    Flatten(),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax') # softmax used for output layer of clustering systems
], name = name)

name = '2_Layer'
cnn_2Layer = Sequential([
    Conv2D(32, kernel_size=3, activation='relu', input_shape=imgShape, name='Conv2D-1'),
    MaxPooling2D(pool_size=2, name='MaxPool'),
    Dropout(0.2, name='Dropout-1'),

    Conv2D(64, kernel_size=3, activation='relu', name='Conv2D-2'),
    Dropout(0.25, name='Dropout-2'),

    Flatten(name='flatten'),

    Dense(64, activation='relu', name='Dense'),
    Dense(10, activation='softmax', name='Output')
], name=name)

name='3_layer'
cnn_3Layer = Sequential([
    Conv2D(32, kernel_size=3, activation='relu', 
           input_shape=imgShape, kernel_initializer='he_normal', name='Conv2D-1'),
    MaxPooling2D(pool_size=2, name='MaxPool'),
    Dropout(0.25, name='Dropout-1'),

    Conv2D(64, kernel_size=3, activation='relu', name='Conv2D-2'),
    Dropout(0.25, name='Dropout-2'),

    Conv2D(128, kernel_size=3, activation='relu', name='Conv2D-3'),
    Dropout(0.4, name='Dropout-3'),

    Flatten(name='flatten'),

    Dense(128, activation='relu', name='Dense'),
    Dropout(0.4, name='Dropout'),
    Dense(10, activation='softmax', name='Output')
], name=name)

# Adds padding and doubles number of filters 32 => 64
# From article written by Jason Brownlee (see Readme)
name='brownlee'
cnn_brownlee = Sequential([
	Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', input_shape=imgShape),
	MaxPooling2D((2, 2)),
	Flatten(),
	Dense(100, activation='relu', kernel_initializer='he_uniform'),
	Dense(10, activation='softmax')
], name = name)

# Brownlee model with a single dropout layer of 0.25
name='brownlee_D25'
cnn_brownlee_D25 = Sequential([
	Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', input_shape=imgShape),
	MaxPooling2D((2, 2)),
    Dropout(0.25, name='Dropout'),
	Flatten(),
	Dense(100, activation='relu', kernel_initializer='he_uniform'),
	Dense(10, activation='softmax')
], name = name)

# Brownlee model with a single dropout layer of 0.50
name='brownlee_D50'
cnn_brownlee_D50 = Sequential([
	Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', input_shape=imgShape),
	MaxPooling2D((2, 2)),
    Dropout(0.50, name='Dropout'),
	Flatten(),
	Dense(100, activation='relu', kernel_initializer='he_uniform'),
	Dense(10, activation='softmax')
], name = name)

# Brownlee combined with 3 Layer model
name='3_brownlee'
cnn_3brownlee = Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', input_shape=imgShape),
    MaxPooling2D(pool_size=2, name='MaxPool'),
    Dropout(0.25, name='Dropout-1'),

    Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', input_shape=imgShape),
    Dropout(0.25, name='Dropout-2'),

    Conv2D(128, kernel_size=3, activation='relu', name='Conv2D-3'),
    Dropout(0.4, name='Dropout-3'),

    Flatten(name='flatten'),

    Dense(128, activation='relu', name='Dense'),
    Dropout(0.4, name='Dropout'),
    Dense(10, activation='softmax', name='Output')
], name=name)


name='4_layer'
cnn_4Layer = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=imgShape),
    BatchNormalization(),

    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    Dropout(0.25),

    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),

    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    Dense(10, activation='softmax')
], name=name)