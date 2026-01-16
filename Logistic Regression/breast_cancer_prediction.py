from sklearn.datasets import load_breast_cancer
import numpy as np

np.random.seed(42)

data = load_breast_cancer()
data_length = data.data.shape[0]
indices = np.random.permutation(data_length)

train_idx = indices[:int(0.8 * data_length)]
test_idx = indices[int(0.8 * data_length):]

X_data = data.data
Y_data = data.target

X_data = np.c_[np.ones((X_data.shape[0], 1)), X_data]

train_features = X_data[train_idx]
train_target = Y_data[train_idx]
test_features = X_data[test_idx]
test_target = Y_data[test_idx]

theta = np.zeros(X_data.shape[1])

mean = np.mean(train_features[:, 1:], axis=0)
std = np.std(train_features[:, 1:], axis=0)

train_features[:, 1:] = (train_features[:, 1:] - mean) / std
test_features[:, 1:] = (test_features[:, 1:] - mean) / std

learning_rate = 0.2
prev_cost = float('inf')
number_of_iterations = 2000
tolerance = 0.0001
train_length = len(train_target)

for i in range(number_of_iterations):
    z = train_features @ theta
    z = np.clip(z, -500, 500)
    train_prediction = 1 / (1 + np.exp(-z))
    train_cost = (-1 / train_length) * np.sum(
        train_target * np.log(train_prediction + 1e-9) +
        (1 - train_target) * np.log(1 - train_prediction + 1e-9)
    )
    if abs(prev_cost - train_cost) < tolerance:
        print("Convergence achieved at:", i)
        break
    gradient = (1 / train_length) * (train_features.T @ (train_prediction - train_target))
    theta -= learning_rate * gradient
    prev_cost = train_cost


print("Cost:", train_cost)

test_z = test_features @ theta
test_z =np.clip(test_z, -500,500)
test_prediction= 1/(1+ np.exp(-test_z))

class_labels= (test_prediction>=0.5).astype(int)

print(class_labels)


accuracy = np.mean(class_labels == test_target)

true_positive = np.sum((class_labels == 1) & (test_target == 1))
false_positive = np.sum((class_labels == 1) & (test_target == 0))
false_negative = np.sum((class_labels == 0) & (test_target == 1))

precision = true_positive / (true_positive + false_positive + 1e-9)
recall = true_positive / (true_positive + false_negative+ 1e-9)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
