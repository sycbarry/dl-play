## overview notes

- If we try and classify and output by `N` classes, the output layer should end with a `Dense` layer of size `N`
- In a single-label, multiclass classification problem, the model ends with  a `softmax` activation so that it will output a probability distribution over the N output classes.
- Use categorical cross_entropy for these kind of problems. It minimizes the prob distribution output of the model and the true distribution of the targets.
- We can handle encoding labels in two ways prior to passing the data into the model.
    (1) Encoding the labels via categorical encoding (on-hot encoding) and using `categorical_crossentropy` as a loss function.
    (2) Encoding the labels as integers and using the `sparse_categorical_cross-entropy` loss function.
- If there are a large amount of categories, we should avoid information bottlenecks in the model by increasing the size of each layer.

