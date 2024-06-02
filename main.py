import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

class CustomSVM:
    def __init__(self, etha=0.01, alpha=0.1, epochs=200, lambda_param=0.01):
        self._epochs = epochs
        self._etha = etha
        self._alpha = alpha
        self._lambda = lambda_param
        self._w = None
        self.history_w = []
        self.train_errors = None
        self.val_errors = None
        self.train_loss = None
        self.val_loss = None

    def add_bias_feature(self, X):
        return np.hstack((X, np.ones((X.shape[0], 1))))

    def fit(self, X_train, Y_train, verbose=False):
        if len(set(Y_train)) != 2:
            raise ValueError("Number of classes in Y is not equal 2!")
        
        X_train = self.add_bias_feature(X_train)
        self._w = np.random.normal(loc=0, scale=0.05, size=X_train.shape[1])
        self.history_w.append(self._w)
        train_errors = []
        train_loss_epoch = []

        for epoch in range(self._epochs):
            tr_err = 0
            tr_loss = 0
            for i in range(X_train.shape[0]):
                margin = Y_train[i] * np.dot(self._w, X_train[i])
                if margin >= 1:
                    self._w -= self._etha * (2 * self._lambda * self._w)
                else:
                    self._w += self._etha * (Y_train[i] * X_train[i] - 2 * self._lambda * self._w)
                    tr_err += 1
                tr_loss += self.soft_margin_loss(X_train[i], Y_train[i])
            if verbose:
                print(f'Epoch {epoch}. Errors={tr_err}. Mean Hinge Loss={tr_loss}')
            train_errors.append(tr_err)
            train_loss_epoch.append(tr_loss)
        
        self.history_w = np.array(self.history_w)
        self.train_errors = np.array(train_errors)
        self.train_loss = np.array(train_loss_epoch)

    def predict(self, X):
        y_pred = []
        X_extended = self.add_bias_feature(X)
        for i in range(len(X_extended)):
            y_pred.append(np.sign(np.dot(self._w, X_extended[i])))
        return np.array(y_pred)
    
    def hinge_loss(self, x, y):
        return max(0, 1 - y * np.dot(x, self._w))
    
    def soft_margin_loss(self, x, y):
        return self.hinge_loss(x, y) + self._alpha * np.dot(self._w, self._w)

    def plot_decision_boundary(self, X, Y, is_linearly_separable):
        X_extended = self.add_bias_feature(X)
        colors = ['red' if label == -1 else 'green' for label in Y]
        plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.7)


        slope = -self._w[0] / self._w[1]
        intercept = -self._w[-1] / self._w[1]
        x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        y_vals = slope * x_vals + intercept
        plt.plot(x_vals, y_vals, 'k--')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Decision Boundary')
        equation_text = f'{self._w[0]:.2f}*x1 + {self._w[1]:.2f}*x2 + {self._w[2]:.2f} = 0'
        plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
        
        if not is_linearly_separable:
            plt.text(0.05, 0.85, f'Data is not linearly separable\nλ = {self._lambda}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
        
        plt.show()

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
    with open(filename, 'r') as file:
        lines = file.readlines()
        N = int(lines[0])
        X = []
        Y = []
        for line in lines[1:]:
            data = line.strip().split()
            x1, x2 = map(float, data[:2])
            label = int(data[2])
            X.append([x1, x2])
            Y.append(label)
    return np.array(X), np.array(Y)


def main():
    input_method = input("Choose input method (manual / file / random): ").strip().lower()
    if input_method == 'manual':
        X, Y = manual_input()
    elif input_method == 'file':
        filename = input("Enter filename: ").strip()
        X, Y = file_input(filename)
    elif input_method == 'random':
        N = int(input("Enter number of data points: "))
        X, Y = generate_data(N)
    else:
        print("Invalid input method selected.")
        return
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    svm = CustomSVM()
    svm.fit(X, Y, verbose=True)
    
    Y_pred = svm.predict(X)
    is_linearly_separable = np.array_equal(Y, Y_pred)
    
    
    if is_linearly_separable:
        print("The data is linearly separable.")
        print(f"Equation of the separating hyperplane: {svm._w[0]:.2f} * x1 + {svm._w[1]:.2f} * x2 + {svm._w[2]:.2f} = 0")
        svm.plot_decision_boundary(X, Y, is_linearly_separable)
    else:
        print("The data is not linearly separable.")
        lambda_param = float(input("Enter the compromise parameter λ: "))
        svm = CustomSVM(lambda_param=lambda_param)
        svm.fit(X, Y, verbose=True)
        print(f"Soft margin solution with λ = {svm._lambda}.")
        print(f"Equation of the decision boundary: {svm._w[0]:.2f} * x1 + {svm._w[1]:.2f} * x2 + {svm._w[2]:.2f} = 0")
        svm.plot_decision_boundary(X, Y, is_linearly_separable)

if __name__ == "__main__":
    main()