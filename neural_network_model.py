import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist # loading datasets from MNIST fashion dataset that contains 60,000 images for training and 10,000 images for validation/testing
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

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

COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
    prediction = model.predict(np.array(image))
    predicted_class = class_names[np.argmax(prediction)]

    show_image(image, class_names[correct_label], predicted_class)
    

def show_image(img, label, guess):
    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.title("Expected: " + label)
    plt.xlabel("Guess: " + guess)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def get_number():
    while True:
        num = input("Pick a number: ")
        if num.isdigit():
            num = int(num)
            if 0 <= num <= 1000:
                return int(num)
        else:
            print("try again...")


num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)