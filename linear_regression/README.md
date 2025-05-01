# notes on scalar regression models

1. We generally use different loss functions for regression. MSE (Mean Squared Error) is a function that we typically use.
2. Evaluation metrics that we use for regression problems revolve around the absolute error, not the accuracy of a prediction. Thus, using MAE (Mean Absolute Error) is the best prediction metric for R models.
3. When we have features in the input data that have different values in different ranges, each feature should be scaled independently as a preprocessing step.
4. When we have little data, use K-fold validation to reliably evaluate a model.
5. When we have littel data, pref to use a small model with few intermediary layers, in order to avoid overfitting.
