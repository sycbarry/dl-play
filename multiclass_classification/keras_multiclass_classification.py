from tensorflow.keras.datasets import reuters
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# load the data to include the 10000 most frequently occuring words in the data set.
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
    num_words=10000
)


# the data is just an index of each word from a dictionary
def reverse_word_index():
    word_index = reuters.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_newswire = " ".join(
        [reverse_word_index.get(i - 3, "?") for i in train_data[0]]
    )
    return decoded_newswire


"""
encoding the input data
- we can either cast the input data as a integer tensor
(for this, we should use the spares_categorical_crossentropy loss function in our model)
- or use a one-hot encoding system that takes in the categorical data which 
embeds each label as an all-zero vector with a 1 in place of the label indx. 
(for this we should use the categorical_crossentropy loss function for our model)
"""

dim = 10000


def vectorize_sequences(sequences, dimension=dim):
    # make the results vector (all 0's)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.0
    return results


# this is a function that we built that can do this for us.
# when using this, make sure to use the categorical_crossentropy loss function for our model
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.0
    return results


# alternatively we can also do this using the builtin functions.
from tensorflow.keras.utils import to_categorical

# one-hot encode the categorical data labels
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

"""
building the model 
- in a `Dense` layer, information is passed down sequentially. to each layer, the only known universe of data is accessible 
from the output of the previous layer. 
- if one layer drops information during the training process, then that information is lost from the next layer. 
- if there are n number of classes, ensure that we are increasing the number of units in each layer as a way to minimize under
capacity of information per layer. 
"""

model = keras.Sequential(
    [
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        # the output is a vector of 46 units long. this is an encoded vector that we have to decode
        # to obtain the literal information that we want.
        # softmax:
        # - the softmax activation function means that each unit in the output vector will be a probability (0, 1).
        # - all the units in the output will sum up to 1
        # - each unit is a probability that matches the likelyhood that the input data matches a previously one-hot encoded classified vector
        layers.Dense(
            46, activation="softmax"
        ),  # note that 46 is the literal number of possible prediction classes in this problem.
    ]
)

# we use the categorical_crossentropy loss function to measure hte distance b/w the probability distributions.
# (this is good to use when we have one-hot encoded values being used as input to the model)
# the prob distribution output by the model and the true distribution of the labels.
# by minimizing the distance b/w the two distributions we can get a true approximation.
model.compile(
    # when using one-hot encoded values, we should use categorical_crossentropy
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# validation dataset
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

"""
training the model 
"""
result = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val),
)


def plot_loss(model_results):
    history = model_results.history
    loss = history["loss"]
    val_loss = history["val_loss"]
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


import sys

val_loss = result.history["val_loss"]
train_loss = result.history["loss"]
sys.stdout.write(f"the training loss: {train_loss}\n")
sys.stdout.write(f"the validation loss: {val_loss}\n")

# plot_loss(result)


"""
Improving our model by reducing the number of epochs.
"""
model = keras.Sequential(
    [
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(46, activation="softmax"),
    ]
)
model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)
model.fit(x_train, y_train, epochs=9, batch_size=512)
results = model.evaluate(x_test, y_test)
# the results from this (on average) reaches around a 80% accuracy rate. which is not bad.
print(results)

predictions = model.predict(x_test)
print(predictions[0].shape)
# >> (46,) ~ this is the 46 unit output layer prediction

print(f"output sum of vector 0: {np.sum(predictions[0])}")
# >> 1 ~ all of the predictions in the output vector sum up to 1 as expected
print(f" the highest score for the 0th index vector: {np.max(predictions[0])}")

sys.stdout.write("\n")
sys.stdout.write("\n")

"""
we can test this against a baseline that we know will achieve a worse result
- in the case that we wanted to use a simple binary classifiation (1 or 0), our model would reach an accuracy score of 
around 50%. 
- below we can find the result of doing a random check (worse possible baseline) and seeing the mean accuracy results
"""

import copy

test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hits_array = np.array(test_labels) == np.array(test_labels_copy)
mean = hits_array.mean()
print(mean)
