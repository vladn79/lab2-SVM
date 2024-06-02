import numpy as np
from sklearn.datasets import make_classification
from input_methods import generate_data


def save_data_to_file(X, Y, filename='data.txt'):
    with open(filename, 'w') as file:
        for i in range(len(X)):
            file.write(f"{X[i][0]} {X[i][1]} {Y[i]}\n")

if __name__ == "__main__":
    N = 2000  # Змініть кількість даних за потреби
    X, Y = generate_data(N)
    save_data_to_file(X, Y)
