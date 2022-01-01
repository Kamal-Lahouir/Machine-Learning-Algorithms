import numpy as np
import matplotlib.pyplot as plt
class Adalin:
    def __init__(self,w0,T_max=100) :
        self.w0 = w0
        self.T_max = T_max
        self.activation_function = self.activation_func
        self.weights = None
        self.loss = []

    def activation_func(self,Y):
         
        #Y : Predictions 
        
        return np.where(Y>0,1,-1)

    def quadratic_error(self,X,Y,W):
          
        # X : Array of simple i
        # W : Weights
        
        n = len(X)
        sum = 0
        for i in range(n):
            sum += (Y[i]-np.vdot(W,X[i]))**2
        return sum/n
    def fit(self,X,Y):
        Y = self.activation_func(Y)
        n_simples,_ = X.shape
        self.weights = self.w0
        
        for t in range(self.T_max):
            for i in range(n_simples) :
                ei = Y[i]-self.activation_func(np.vdot(self.weights,X[i]))
                if ei != 0:
                    self.weights += 2*ei*X[i]
            Ls = self.quadratic_error(X,Y,self.weights)
            self.loss.append([t,Ls])
        return self.weights
    def predict(self,X):

        # X: array of test set of samples

        output = np.array([np.vdot(x,self.weights) for x in X])
        return self.activation_func(output)

    def accuracy(self,Y_predicted,Y_test):
        Y_predicted = self.activation_func(Y_predicted)
        Y_test      = self.activation_func(Y_test)
        acc = 0
        for i in range(len(Y_predicted)):
            if Y_predicted[i] == Y_test[i]:
                acc += 1
        return acc/len(Y_test)

    def draw_loss(self):
        x = np.array(self.loss)[:,0]
        y = np.array(self.loss)[:,1]
        plt.ylabel("Emperical Error")
        plt.xlabel("Iterations")
        plt.plot(x,y)
        plt.show()