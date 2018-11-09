from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============= Load the data  ==============
boston_housing = keras.datasets.boston_housing
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# Shuffle the training set
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

# check the data
print("Training set: {}".format(train_data.shape))  # 404 examples, 13 features
print("Testing set:  {}".format(test_data.shape))   # 102 examples, 13 features


# ============= Use pandas to display the first few rows of dataset  ==============
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']

df = pd.DataFrame(train_data, columns=column_names)
print(df.head())
print("Train data are not normalized here. They have different scales")
print("train_labels(house prices): {}".format(train_labels[0:10]))

# ============= Pre-process: Normalize data  ==============
print("\n For each feature, subtract the mean of the feature and divide by the standard deviation.\n")
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

print("\n First training sample, normalized.\n")
df = pd.DataFrame(train_data, columns=column_names)
print(df.head())

# ============= Build the Neural Network model ==============
model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(train_data.shape[1],)),  #Hidden layer 1
    keras.layers.Dense(64, activation=tf.nn.relu),  #Hidden layer 2
    keras.layers.Dense(1)  #Output layer
                         ])
optimizer = tf.train.RMSPropOptimizer(0.001)  # Optimizer that implements the RMSProp algorithm
model.compile(loss='mse',
              optimizer=optimizer,
              metrics=['mae'])   # A metric function is similar to a loss function,
                                 # except the results from evaluating a metric are not used when training the model.
                                 # Mean Absolute Error (MAE)
print(model.summary())

# ====================== Train the model =======================
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 500

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

# Store training stats
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])

# ====================== Plot the results =======================
def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [1000$]')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([0, 5])
  plt.show()

plot_history(history)


# ============= Test or Evaluate the model ==============
[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
print("\n Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))


# ============= Use the trained model ==============
test_predictions = model.predict(test_data).flatten()
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [1000$]')
plt.ylabel('Predictions [1000$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
plt.plot([-100, 100], [-100, 100])
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins = 50)
plt.xlabel("Prediction Error [1000$]")
plt.ylabel("Count")
plt.show()




