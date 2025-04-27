## some notes on binary classification

1. Need to do a bit of preprocessing on the data prior to feeding it into a nn.
2. Sequences of words can be encoded as binary vectors (so can integers)
3. Stacks of `Dense` layers with a `relu` activation function is generally widely used for a wide range of problems. (sentiment classification)
4. For Binary Classification problems, the model should end with a `Dense` layer and a `sigmoid` afunction. The output of the model should be scalar to encode a prob.
5. With scalar sigmoid output on a binary classification problem, use the `binary_crossentropy` function
6. `rmsprop` optimizer is good for a wide range of problems.
7. Avoid overfitting, and ensure to plot scores (accuracy and loss) so that we can tune models after training.
