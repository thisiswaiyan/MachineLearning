import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.layers import BatchNormalization

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.45))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_images, train_labels, batch_size=128, epochs=10, verbose=1, validation_data=(test_images, test_labels))

score = model.evaluate(test_images, test_labels, verbose=0)
print(f"Test loss: {score[0]}")
print(f"Test accuracy: {score[1]}")

import cv2
import matplotlib.pyplot as plt

local_image = cv2.imread("ZERO.jpg")
gray_image = cv2.cvtColor(local_image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_image,cmap='Greys')
plt.show()
print(gray_image.shape)
resize_image = cv2.resize(gray_image, (28, 28))
print(resize_image.shape)
reshape_image = np.reshape(resize_image, (28, 28, 1))
print(reshape_image.shape)

x = np.expand_dims(reshape_image, axis=0)
preds = model.predict(x)
predict_class = np.argmax(preds)
print(preds[0])
print(predict_class)
