import numpy as np
import matplotlib.pyplot as plt

class logisticRegression:

    def __init__(self, lr = 0.001, n_iters = 1000):
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
            #Computation 
            linear_model = np.dot(X_, self.Theta)
            y_predicted = self._sigmoid(linear_model)
            D = (y_predicted - y)
            # Computation of the loss
            np.seterr(divide = 'ignore') 
            J = -(1/n_samples)*(np.dot(y.T, np.log(y_predicted)+np.dot((1-y).T, np.log(1-y_predicted))))
            
            self.loss.append([_,J])
            print(self.loss[_])
            # Computation of the gradient
            dw = (1/n_samples)*np.dot(X_.T, D)
            # Update of theta
            self.Theta -= (self.lr * dw)
        return self.Theta

    def predict(self, X):
        X_ = np.hstack((X,np.ones((len(X),1))))
        linear_model = np.dot(X_, self.Theta)
        y_predicted = self._sigmoid(linear_model)
        y_predicted_clear = [ 1 if x > 0.5 else 0 for x in y_predicted]
        return y_predicted_clear

    def accuracy(self,y_true,y_predict):
        accuracy = np.sum(y_true == y_predict) / len(y_true)
        return accuracy

    def _sigmoid(self, X):
        return 1/(1+np.exp(-X))

    def draw_loss(self):
        x = np.array(self.loss)[:,0]
        y = np.array(self.loss)[:,1]
        plt.ylabel("Emperical Error")
        plt.xlabel("Iterations")
        plt.plot(x,y)
        plt.show()