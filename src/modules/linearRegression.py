import numpy as np
import matplotlib.pyplot as plt

class linearRegression:

    def __init__(self, lr = 0.01, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.Theta = None #Theta includes weights and the bias
        self.loss = []

    def fit(self, X, y):
        print(X.shape)
        n_samples , n_features = X.shape
        X_ = np.hstack((X,np.ones((len(X),1))))
        self.Theta = np.zeros(n_features+1)
        for _ in range(self.n_iters):
            D = (np.dot(X_, self.Theta)- y)
            # Computation of the loss
            J = (1/2*n_samples)*np.sum(D**2)
            self.loss.append([_,J])
            # Computation of the gradient
            dw = (1/n_samples)*np.dot(X_.T, D)
            # Update of theta
            self.Theta -= (self.lr * dw)
        return self.Theta

    def predict(self, X):
        X_ = np.hstack((X,np.ones((len(X),1))))
        return np.dot(X_, self.Theta)

    def accuracy(self,X,y):
        print(self.Theta.shape)
        X_ = np.hstack((X,np.ones((len(X),1))))
        u = np.sum((np.dot(X_,self.Theta)-y)**2) 
        v = np.sum((y.mean()-y)**2)
        return 1-u/v

    def draw_loss(self):
        x = np.array(self.loss)[:,0]
        y = np.array(self.loss)[:,1]
        plt.ylabel("Emperical Error")
        plt.xlabel("Iterations")
        plt.plot(x,y)
        plt.show()