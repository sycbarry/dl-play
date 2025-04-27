"""
Simple Linear Classifier 
- this just tries to find the parameters of a line that neatly seperates two classes of data 
"""


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys

"""
Set up the data, including the input and the target range
"""
# build out some synthetic data
num_samples_per_class = 1000
negative_samples = np.random.multivariate_normal(
    mean=[0, 3], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class
)
positive_samples = np.random.multivariate_normal(
    mean=[3, 0], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class
)

# np vstack stacks arrays vertically, in a row wise manner. it takes sequence of arrays is input and returns a
# single array formed by stacking the input arrays on top of each other.
inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)

targets = np.vstack(
    (
        np.zeros((num_samples_per_class, 1), dtype="float32"),
        np.ones((num_samples_per_class, 1), dtype="float32"),
    )
)


"""
Set up the model
"""
input_dim = 2
output_dim = 1

W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(
    initial_value=tf.zeros(
        shape=output_dim,
    )
)


def model(inputs):
    return tf.matmul(inputs, W) + b


def square_loss(targets, predictions):
    """
    this is the msqe loss function
    """
    per_sample_loss = tf.square(targets - predictions)
    # this averages the per-sample loss scores into a single scalar loss value
    return tf.reduce_mean(per_sample_loss)


# set the learning rate.
learning_rate = 0.1


def training_step(inputs, targets):
    # the tf.GradientTape is a context manager that records operations on tensors.
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = square_loss(predictions, targets)
    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])
    W.assign_sub(grad_loss_wrt_W * learning_rate)
    b.assign_sub(grad_loss_wrt_b * learning_rate)
    return loss


"""
-----
Training
-----
here we'll do batch training.
we run each training step for all the data, rather than iterate over the data in small batches. 
each training step takes longer because we do forward passes and gradient adjustments for all the records in the data set.
this is more efficient for training however, because all training records are maintained in the training process
"""

epochs = 40

for step in range(epochs):
    loss = training_step(inputs, targets)
    sys.stdout.write(f"Loss at step: {step}: loss: {loss:.4f}\n")

"""
predictions = model(inputs)
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()
"""

predictions = model(inputs)
x = np.linspace(-1, 4, 100)
y = -W[0] / W[1] * x + (0.5 - b) / W[1]  # this is the line's equation.
plt.plot(x, y, "r")
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()
