import numpy as np
import matplotlib.pyplot as plt

class PolynomialRegression:
    def __init__(self, k_ord = 2, lr = 0.1, X , Y, max_iterations= 100 ):
        self.k_ord = k_ord
        self.lr = lr
        self.max_iterations = max_iterations
        self.predict_function = self.predict_fct
        self.gradient_function = self.mean_squared_gradient
        self.weights = np.array([])

        for i in range(2, self.k_ord + 1):
            X_inp = np.array([np.concatenate((x, [x[0] ** i])) for x in X]) 

        self.X = np.array([np.concatenate((x, np.array([1]))) for x in X_inp])

    
    def mean_squared_gradient(self, X, Y):
        gradients = []

        for j in range((X).shape[1]):
            gradient = 0

            for i in range(X.shape[0]):
                x = X[i]

                gradient += x[j] * (self.predict_function(x) - Y[i])

            gradient *= (2.0 / X.shape[0])
            gradients.append(gradient)

        return np.array(gradients)
    
    def fit(self):
        # WE compute our hypothesis first
        for i in range(self.max_iterations):
            gradients = self.gradient_function(self.X, self.Y)
            self.weights = self.weights - self.lr * gradients


    def predict_fct(self,):
        pass
    def accuracy(self,):
        pass

