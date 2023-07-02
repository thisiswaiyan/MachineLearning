import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist # loading datasets from MNIST fashion dataset that contains 60,000 images for training and 10,000 images for validation/testing
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#print(train_images.shape)
#print(train_images[0, 23, 23])
#print(train_labels[:10]) # first 10 training labels

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

#plt.figure()
#plt.imshow(train_images[1])
#plt.colorbar()
#plt.grid(False)
#plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), # input layer 
    keras.layers.Dense(128, activation='relu'), # hidden layer
    keras.layers.Dense(10, activation='softmax') # output layer
])

model.compile(optimizer='adam', # function for gradient descent
              loss='sparse_categorical_crossentropy', # cost function
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=4)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print("Test accuracy: ", test_acc)

predictions = model.predict(test_images)
print(class_names[np.argmax(predictions[78])])
plt.figure()
plt.imshow(test_images[78])
plt.colorbar()
plt.grid(False)
plt.show()