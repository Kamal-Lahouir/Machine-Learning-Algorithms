import numpy as np
import matplotlib.pyplot as plt
# Class of the model

class Perceptron:
    def __init__(self,eps=0.1,n_iters=1000, lr = 0.1) :
        self.n_iters = n_iters
        self.eps = eps
        self.activation_function = self.activation_func
        self.weights = None
        self.learning_rate = lr
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
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.random.rand(n_features)

        y = self.activation_func(Y)

        Ls = self.emperical_error(X,Y)

        for _ in range(self.n_iters):

            for idx, x_i in enumerate(X):

                y_predicted = self.predict(x_i)

                # Perceptron update rule
                update = self.learning_rate * (y[idx] - y_predicted)

                Ls = self.emperical_error(X,Y)
                self.loss.append([idx,Ls])
                x_i.reshape(3,3)
                
                self.weights += update * x_i
        return self.weights

    # Predction function: it predict a value of an input through the model and not the y of the data
    def predict(self,X):
        # X: array of test set of samples
        output = np.array([np.dot(x,self.weights) for x in X])
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
