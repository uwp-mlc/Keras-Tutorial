import numpy

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

seed = 7
numpy.random.seed(seed)

# load data from mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# flatten 28*28 images to a 784 vector for each image
num_pixels = x_train.shape[1] * x_train.shape[2]

x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')

# normalize inputs from 0-255 to 0-1
x_train = x_train / 255
x_test = x_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, 
        input_dim=num_pixels, 
        activation='relu'))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # compile the model
    model.compile(loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

    return model

# build the model
model = baseline_model()

# train the model
model.fit(x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=10,
    batch_size=200,
    verbose=2)


# import matplotlib.pyplot as plt

# plt.subplot(221)
# plt.imshow(x_train[0], cmap=plt.get_cmap('gray'))
# print(y_train[0])
# plt.show()