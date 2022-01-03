
import numpy as np
import matplotlib.pyplot as plt
# Class of the model

class Perceptron:
    def __init__(self,eps=0.1,n_iters=1000) :
        self.n_itrs = n_iters
        self.eps = eps
        self.activation_function = self.activation_func
        self.weights = None
        self.loss = []
    
    def sign(self,W,X):
        
        # X : Array of simple i
        # W : Weights
        
        return np.sign(np.vdot(W,X))


    # Activation function
    def activation_func(self,Y):
        
        #Y : Predictions 
        
        return np.where(Y>0,1,-1)
    

    # Unit function: returns 1 if a is different than b or 0 otherwise
    def unit_func(self,a,b):
        return 1 if a!=b else 0

    # Function that computes the empirical error of the model
    def emperical_error(self,X,Y):
        
        # X : array of all simples where the rows are the simples
        # Y : array of predictions
        n = len(X)
        sum = 0
        for i,x_i in enumerate(X) :
            sum += self.unit_func(self.activation_func(np.vdot(self.weights,x_i)),Y[i])
        return sum/n

    # Fit function with which we train our model
    def fit(self,X,Y):

        y = self.activation_func(Y)
        n_simples,n_features = X.shape
        self.weights = np.random.rand(n_features)
        
        Ls = self.emperical_error(X,Y)
        t = 0
        while (Ls > self.eps) :
            for i in range(n_simples):
                if self.sign(self.weights,X[i])*Y[i] <= 0:
                    self.weights += Y[i]*X[i]
                    t += 1
                    Ls = self.emperical_error(X,Y)
                    self.loss.append([t,Ls])
        return self.weights

    # Predction function: it predict a value of an input through the model and not the y of the data
    def predict(self,X):
        # X: array of test set of samples
        output = np.array([np.vdot(x,self.weights) for x in X])
        return self.activation_func(output)

    # Accuracy function: it returns the accuracy of the prediction in our model
    def accuracy(self,Y_predicted,Y_test):
        Y_predicted_ = self.activation_func(Y_predicted)
        Y_test_      = self.activation_func(Y_test)
        acc = 0
        for i in range(len(Y_predicted_)):
            if Y_predicted_[i] == Y_test_[i]:
                acc += 1
        return acc/len(Y_test_)

    # Loss drawing function: it gives the shape of the loss function through the list self.loss
    def draw_loss(self):
        X = np.array(self.loss)[:,0]
        Y = np.array(self.loss)[:,1]
        plt.ylabel("Emperical Error")
        plt.xlabel("Iterations")
        plt.plot(X,Y)
        plt.show()
