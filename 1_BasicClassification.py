## =========== TRANSCRIPT ================
# 1- import stuff
# 2-


## =======================================
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
from time import time

# ============= Load the data  ==============
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# ============= Create class names ==============
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ============= What are the data ==============
print(train_images.shape)  # Each label is an integer between 0 and 9
print(train_labels)

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()  # This should be used in the IDE=Pycharm to show the plot

# ============= Pre-process the data ==============
# We scale these values to a range of 0 to 1 before feeding to the neural network model

train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# ============= Build the Neural Network model ==============
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # Layer0 input: transforms the images 28*28, to a 1d-array of 784*1 pixels
    keras.layers.Dense(128, activation=tf.nn.relu), # Layer1 = keras.layers.Dense = 128 densely-connected, or fully-connected neurons
    keras.layers.Dense(10, activation=tf.nn.softmax) # Output Layer = 10 fully-connected neurons => 10 probabilities
])

# Activation for hidden layers are usually "ReLu"
# Activation for hidden layers are usually "softmax"

# ============= Compile or Prepare the Neural Network model ==============
model.compile(optimizer=tf.train.AdamOptimizer(),  # Optimizer function
              loss='sparse_categorical_crossentropy', # define the loss function
              metrics=['accuracy'])  # the fraction of the images that are correctly classified.

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

# ============= Train the model ==============
model.fit(train_images, train_labels, epochs=5, callbacks=[tensorboard])  # 5 time the whole train dataset is used

# ============= Test or Evaluate the model ==============
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_accuracy)


# ============= Use the trained model ==============
predictions = model.predict(test_images) # or use another evaluation set
print(predictions[0]) #probabilities for each class
print(np.argmax(predictions[0])) #print the class with highest probability

print(class_names[np.argmax(predictions[0])]) #print the class with highest probability


# ============= Define Functions to plot ==============

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.get_cmap('gray'))

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]), color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    xrange = np.arange(10)
    plt.xticks(xrange, class_names, rotation='vertical')
    plt.yticks(predictions_array)
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


i = 91
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()


