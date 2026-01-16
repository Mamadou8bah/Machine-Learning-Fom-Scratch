import pandas as pd
import numpy as np


student_performance = pd.read_csv('student-mat.csv', sep=';')
all_features=student_performance.iloc[:,:-1]
target = student_performance.iloc[:, -1:].to_numpy().flatten()

binary_maps = {
    "school": {"GP": 1, "MS": 0},
    "sex": {"M": 1, "F": 0},
    "address": {"U": 1, "R": 0},
    "famsize": {"GT3": 1, "LE3": 0},
    "Pstatus": {"T": 1, "A": 0},
    "schoolsup": {"yes": 1, "no": 0},
    "famsup": {"yes": 1, "no": 0},
    "paid": {"yes": 1, "no": 0},
    "activities": {"yes": 1, "no": 0},
    "nursery": {"yes": 1, "no": 0},
    "higher": {"yes": 1, "no": 0},
    "internet": {"yes": 1, "no": 0},
    "romantic": {"yes": 1, "no": 0},
}

one_hot_categories = {
    "Mjob": ["teacher", "health", "services", "at_home"],
    "Fjob": ["teacher", "health", "services", "at_home"],
    "reason": ["home", "reputation", "course"],
    "guardian": ["mother", "father"],
}

numeric_cols = [
    "age", "Medu", "Fedu", "traveltime", "studytime",
    "failures", "famrel", "freetime", "goout",
    "Dalc", "Walc", "health", "absences"
]


def encode_row(row):
    features = []

    # bias
    features.append(1)

    # numeric features
    for col in numeric_cols:
        features.append(row[col])

    # binary categorical
    for col, mapping in binary_maps.items():
        features.append(mapping[row[col]])

    # one-hot categorical
    for col, categories in one_hot_categories.items():
        for cat in categories:
            features.append(1 if row[col] == cat else 0)

    return np.array(features, dtype=float)

X = np.vstack([encode_row(row) for _, row in all_features.iterrows()])



theta=np.zeros(X.shape[1])


np.random.seed(42)

m=X.shape[0]

indices=np.random.permutation(m)

train_idx=indices[:int(0.8*m)]
test_idx=indices[int(0.8*m):]

train_features=X[train_idx]
test_features=X[test_idx]
train_target=target[train_idx]
test_target = target[test_idx]

train_prediction= train_features @ theta;

train_cost = (1/(2*train_features.shape[0])) * np.sum((train_prediction - train_target)**2)


prev_train_cost=float("inf");
learning_rate=0.001

print("Previous train cost: ",train_cost)
for i in range(2000):
    train_prediction= train_features @ theta
    train_cost = (1/(2*train_features.shape[0])) * np.sum((train_prediction - train_target)**2)
    if abs(prev_train_cost-train_cost)<0.000001:
        break
    gradient_descent = (1/train_features.shape[0]) * (train_features.T @ (train_prediction - train_target))

    theta-=learning_rate*gradient_descent;
    prev_train_cost=train_cost


print("Final Train Cost: ",train_cost)

test_prediction= test_features @ theta

test_cost = (1/(2*test_features.shape[0])) * np.sum((test_prediction - test_target)**2)


print("Test cost: ",test_cost)

print("Theta values: ")
print(theta)





