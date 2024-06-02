import numpy as np
from sklearn.preprocessing import StandardScaler
from svm import SVM
from input_methods import manual_input, file_input, generate_data


def main():
    input_method = input("Choose input method (manual / file / random): ").strip().lower()
    if input_method == 'manual':
        X, Y = manual_input()
    elif input_method == 'file':
        filename = 'data.txt'
        X, Y = file_input(filename)
    elif input_method == 'random':
        N = int(input("Enter number of data points: "))
        X, Y = generate_data(N)
    else:
        print("Invalid input method selected.")
        return
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    svm = SVM()
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
        svm = SVM(lambda_param=lambda_param)
        svm.fit(X, Y, verbose=True)
        print(f"Soft margin solution with λ = {svm._lambda}.")
        print(f"Equation of the decision boundary: {svm._w[0]:.2f} * x1 + {svm._w[1]:.2f} * x2 + {svm._w[2]:.2f} = 0")
        svm.plot_decision_boundary(X, Y, is_linearly_separable)

if __name__ == "__main__":
    main()