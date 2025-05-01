"""
This is a simple linear regression script that uses the boston housing dataset 
from keras.
"""

from tensorflow.keras.datasets import boston_housing
import sys
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

"""
This data set contains a slew of different values for each feature. The ranges are wildly different. 
This is typically something to avoid, so we use feature normalization. 

Feature normalization:
    For each feature in the data set we subtract the mean of the feature and divide by the standard deviation, so that 
    the feature is centered around 0 and has a unit standard deviation.

"""

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std


"""
Building the model
(we can generally avoid overfitting by building a smaller model)
"""


def build_scalar_regression_model():
    model = keras.Sequential(
        [
            layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(1),
            # The model ends with a single unit, and no acivation function (linear layer).
            # no activation function in the output layer means that the model can learn to predict any output (1, 65, 2999, etc). It isn't
            # constrained to values b/w 1 or 0 for example.
        ]
    )
    # mse error loss function is generally a good loss function to use for regression problems.
    # mae (as the prediction metric) measures the absolute distance between the prediction and the target. (the lower the value, the better the prediction)
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model


"""
Working with small data sets. (K-Validation)
 
When a data set has a small set of records, we want to avoid high variance.
(especially if we want to split our data further into, test, train, validation sets)

What we can do, is to use something called K-fold cross-validation.
(1) This involves splitting the data into K partitions (usually K=4)
(2) Instantiating K identical models
(3) Training each model on K-1 partitions
(4) Evaluating on the remaining partition.
(5) Using the average of the K validation partition scores obtained as the final score.
"""
k = 4
num_val_samples = len(train_data) // k  # create the partitions
num_epochs = 500
all_scores = []
all_mae_histories = []
for i in range(k):
    sys.stdout.write(f"Processing fold #{i}\n")

    # make the validation data
    val_data = train_data[i * num_val_samples : (i + 1) * num_val_samples]
    # make the ** targets
    val_targets = train_targets[i * num_val_samples : (i + 1) * num_val_samples]

    # prepare the training data and testing data
    partial_train_data = np.concatenate(
        [train_data[: i * num_val_samples], train_data[(i + 1) * num_val_samples :]],
        axis=0,
    )
    partial_train_targets = np.concatenate(
        [
            train_targets[: i * num_val_samples],
            train_targets[(i + 1) * num_val_samples :],
        ],
        axis=0,
    )
    model = build_scalar_regression_model()
    history = model.fit(
        partial_train_data,
        partial_train_targets,
        validation_data=(val_data, val_targets),
        epochs=num_epochs,
        batch_size=16,
        verbose=0,
    )
    mae_history = history.history["val_mae"]
    all_mae_histories.append(mae_history)
    # val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)

    # append the val_mae to our tracker
    # all_scores.append(val_mae)

average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)
]

"""
Plot the figures of each mae score
--


plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()
"""


# truncated plotting
truncated_mae_history = average_mae_history[10:]
plt.plot(range(1, len(truncated_mae_history) + 1), truncated_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()

"""
Notes:
    (1) In the above code, we truncated the average_mae_history to all values past the first 10 records. 
    (2) When we graph the truncated_mae_history values, we see that we start overfitting the data (no improvement to our training)
    (3) What we can do, is to tune our model's parameters to include only the first n epochs prior to where the model starts to overfit. 
        We can also add more intermediary layers to the baseline model.
"""


"""
Let's train it on the entirity of the dataset
"""
model = build_scalar_regression_model()
model.fit(train_data, train_targets, epochs=130, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
sys.stdout.write(f"test_mae_score: {test_mae_score}")

# prediction
predictions = model.predict(test_data)
sys.stdout.write(f"prediction 0 {predictions[0]}")
