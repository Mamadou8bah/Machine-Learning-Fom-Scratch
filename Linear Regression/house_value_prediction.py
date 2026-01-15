from sklearn.datasets import fetch_california_housing
import numpy as np

# Here, I import the California Housing dataset using sklearn.
# This dataset contains housing features and corresponding house prices.
data = fetch_california_housing()

# I set a random seed so that all random operations
# (such as shuffling the data) are reproducible.
np.random.seed(42)

# X represents the feature matrix of the dataset.
# Each row corresponds to one house and each column to one feature.
X = data.data

# Y represents the target values of the dataset,
# which are the median house prices.
Y = data.target

# I add a bias (intercept) term by inserting a column of ones
# as the first column of the feature matrix.
X = np.c_[np.ones((X.shape[0], 1)), X]

# I randomly shuffle the indices of the dataset
# to ensure that the train/test split is random.
indices = np.random.permutation(len(X))

# I define the split index so that 80% of the data
# is used for training and 20% for testing.
split_index = int(0.8 * len(X))

# I select indices for the training and testing sets.
train_idx = indices[:split_index]
test_idx  = indices[split_index:]

# Using the indices, I create the training and testing feature sets
# and their corresponding target values.
train_features = X[train_idx]
train_target   = Y[train_idx]
test_features  = X[test_idx]
test_target    = Y[test_idx]

# I determine the number of features (including the bias term).
number_of_features = train_features.shape[1]
print(number_of_features)

# I initialize the parameter vector theta with zeros.
# Each value in theta corresponds to one feature weight.
theta = np.zeros(number_of_features)

# I compute the mean and standard deviation of the training features,
# excluding the bias column, in order to normalize the data.
mean = np.mean(train_features[:, 1:], axis=0)
std  = np.std(train_features[:, 1:], axis=0)

# I normalize the training features using the computed mean and standard deviation.
train_features[:, 1:] = (train_features[:, 1:] - mean) / std

# I apply the same normalization parameters to the test features
# to ensure consistency.
test_features[:, 1:]  = (test_features[:, 1:] - mean) / std

# I compute the initial predictions using the hypothesis function:
# h_theta(X) = X @ theta.
train_predictions = train_features @ theta

# I store the number of training examples.
m = len(train_features)

# I compute the initial cost using the mean squared error formula.
train_cost = (1 / (2 * m)) * np.sum((train_predictions - train_target) ** 2)
print(train_cost)

# I initialize the previous training cost to infinity
# so that the convergence check works correctly.
prev_train_cost = float("inf")

# I define the learning rate for gradient descent.
learning_rate = 0.01

# I apply gradient descent to minimize the cost function.
number_of_iterations = 0
for i in range(10000):
    number_of_iterations += 1

    # I compute predictions using the current values of theta.
    train_predictions = train_features @ theta

    # I calculate the current cost.
    train_cost = (1 / (2 * m)) * np.sum((train_predictions - train_target) ** 2)

    # I stop the algorithm if the change in cost is very small,
    # indicating that the model has converged.
    if abs(prev_train_cost - train_cost) < 1e-7:
        break

    # I compute the gradient of the cost function.
    gradient_descent = (1 / m) * (train_features.T @ (train_predictions - train_target))

    # I update the parameters using gradient descent.
    theta -= learning_rate * gradient_descent

    # I update the previous cost for the next iteration.
    prev_train_cost = train_cost

# I print the final training cost after convergence.
print("Final training cost")
print(train_cost)

# I generate predictions for the test dataset.
test_prediction = test_features @ theta

# I store the number of test examples.
m_test = len(test_features)

# I compute the test cost using the mean squared error formula.
print("Test Cost")
test_cost = (1 / (2 * m_test)) * np.sum((test_prediction - test_target) ** 2)
print(test_cost)

# I print the number of iterations required for convergence.
print("Number of iterations", number_of_iterations)

# Finally, I print the feature names and the learned parameters.
# theta[0] represents the bias, while theta[1:] correspond to the features.
print(data.feature_names)
print(theta)
