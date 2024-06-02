import numpy as np

from sklearn.datasets import make_classification

def generate_data(N, random_state=42):
    X, Y = make_classification(n_samples=N, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, flip_y=0, random_state=random_state)
    Y[Y == 0] = -1
    return X, Y

def manual_input():
    N = int(input("Enter number of data points: "))
    X = []
    Y = []
    for i in range(N):
        x1, x2 = map(float, input(f"Enter coordinates for data point {i+1} (x1 x2): ").split())
        label = int(input(f"Enter class label for data point {i+1} (-1 or 1): "))
        X.append([x1, x2])
        Y.append(label)
    return np.array(X), np.array(Y)

def file_input(filename):
    X = []
    Y = []
    with open(filename, 'r') as file:
        for line in file:
            data = line.strip().split()
            x1, x2 = map(float, data[:2])
            label = int(data[2])
            X.append([x1, x2])
            Y.append(label)
    return np.array(X), np.array(Y)