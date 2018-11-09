#======================
# This code classifies movie reviews as positive or negative
# using the text of the review.

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
from time import time


# ===================== Load the data  ========================

# data is preprocessed such that the review (sequences of words) is converted to sequences of integers,
# where each integer represents a specific word in a dictionary

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# keeps the top 10,000 most frequently occurring words in the training data


# ===================== What are the data ====================
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

#  Here's what the first review looks like:
print(train_data[0])


# ================ Convert the integers back to words ================
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(train_data[2]))

# ====================== Prepare the data =========================
# The reviews—the arrays of integers—must be converted to tensors before fed into the neural network

# we  pad the arrays so they all have the same length,
# then create an integer tensor of shape max_length * num_reviews.
# We can use an embedding layer capable of handling this shape as the first layer in our network.

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

# A max length of 256 word is considered. We use zeros to pad the remaining spaces
print(train_data[0]) # This is how a sample review looks like after padding


# ====================== Build the Neural Network model ==================
vocab_size = 10000

# Structure of the NN:
# Input layer: 256 (a review)
# Hidden Layer 1: 16 (each word to a 16d array) + averaging
# Hidden Layer 2: 16
# Output Layer: 1

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))    # The embedding layer is basically a look up table.
                                                     # using the words (int) from the dictionary,
                                                     # it assigns any input with a 16-d array.

model.add(keras.layers.GlobalAveragePooling1D())     #
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

print(model.summary())

# ====================== Compile the Neural Network model ==================
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs3/{}".format(time()))

# ====================== Create validation sets ==================
x_val = train_data[:10000] # Beginning to the 10000th
partial_x_train = train_data[10000:]  # 10001th to end

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# ======================== Train the model ========================
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1,
                    callbacks=[tensorboard])

# ======================== Test the model ========================
results = model.evaluate(test_data, test_labels)
print(results)

# ========== Create a graph of accuracy and loss over time ==========
history_dict = history.history
print(history_dict.keys())

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


#plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()





