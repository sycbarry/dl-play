from tensorflow.keras.datasets import imdb
import sys
import numpy as np


"""
test, train split of data from the imdb dataset
- the data has been split evenly
- words are converted into numbers .
- the labels are either 1, 0 with 1 showing a positive result and a 0 showing a negative result.
"""
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


word_index = imdb.get_word_index()
reversed_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = " ".join([reversed_word_index.get(i - 3, "?") for i in train_data[0]])

"""
multi-hot encode the data
- here we turn the list of integers into n-long vectors, with the digit (say 8) being set to a 1 in the n-long vector at 
index 8 from right to left
- we generally don't want to feed in integers into the neural network, as each integer has a different length 
and makes the incosistency of those length's imcompatible with a accurate neural network output.
- So our goal is to standardize the input in a way that make the lengths consisent with the input's dimensions
"""

dim = 10000


def vectorize_sequences(sequences, dimension=dim):
    # make the results vector (all 0's)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.0
    return results


# this will give us a 'dim' length vector of 1's and 0's
def make_v_train_test_split(train_data, test_data):
    return (vectorize_sequences(train_data), vectorize_sequences(test_data))


x_train, x_test = make_v_train_test_split(train_data, test_data)
sys.stdout.write(f"length of x_train and x_test: {len(x_train)}, {len(x_test)}")

# we also want to vectorize the labels, these are arrays of scalars
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")

"""
a model that behaves well on input data being all vectors and labels being all scalars, is a stack of 
densely connected layers with a relu activation function.
- how many layers to we use in our dnn
- how many units do we want to take in (n -> n -> 1)?
    ie 16 input for the first layer with 16 output. same with the second. the third layer takes in 16 and outputs 1 output.
"""

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential(
    [
        layers.Dense(16, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val),
)


"""
plotting the training and validation loss
(
    For the model we had used so far, our training loss falls close to zero by the 20th epoch...
    however, the validation loss seems to increase, this is not a good thing. It generally means that we are 
    overfitting after about 4 epochs.
)
"""
import matplotlib.pyplot as plt


def plot_training_and_validation_loss(model):
    # get the history of the model's training and evaluation run
    history_dic = model.history

    loss_values = history_dic["loss"]
    val_loss_values = history_dic["val_loss"]
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, "bo", label="Training loss")
    plt.plot(epochs, val_loss_values, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# plot_training_and_validation_loss(history)


"""
Training a new model using 4 epochs instead of 20, to minimize overfitting.
"""
model = keras.Sequential(
    [
        layers.Dense(16, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
model_2_results = model.fit(
    x_train,
    y_train,
    epochs=4,
    batch_size=512,
    validation_data=(x_val, y_val),
)  # note we are decreasing our epoch time to 4 instead of 20

# if we uncomment the function call below, we see that the model_2_results have a much better validation loss
# plot_training_and_validation_loss(model_2_results)

results = model.evaluate(x_test, y_test)
print(results)

# note that the closer the model is to .5 (50%), the less "confident" the model is with its predictions.
# the closer the model's predictions are to 1 or 0, the more confident it is with its predictions
sys.stdout.write("predictions...\n")
print(model.predict(x_test))

"""
Closing notes. 
- we can play around with the model's hyperparameters to see if we can increase the model's performance. 
- we can adjust the representation layers in the model (prior to the output layer) to improve the performance.
- we can use a different loss function in each layer or a different activation function.
"""
