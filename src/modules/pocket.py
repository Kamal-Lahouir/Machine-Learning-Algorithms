
import numpy as np
import matplotlib.pyplot as plt
class Pocket:
    def __init__(self,w0,T_max=100) :
        self.w0 = np.array(w0)
        self.T_max = T_max
        self.activation_function = self.activation_func
        self.weights_t = None # for the PLA update
        self.weight_s    = None # for the pocket update
        self.weights = None
        self.loss = []
    
    def sign(self,W,X):
        '''
        x : an array of one sample
        w : weights
        '''

        return np.sign(np.vdot(W,X))

    def activation_func(self,Y):
        '''
        y : the predections 
        '''
        return np.where(Y>0,1,-1)
    
    def unit_func(self,a,b):
        return 1 if a!=b else 0

    def emperical_error(self,X,Y,W):
        '''
        X : array of all simples (rows are the simples, columns are the features)
        y : array of the predictions
        return the emperical error.
        '''
        n = len(X)
        sum = 0
        for i,x_i in enumerate(X) :
            sum += self.unit_func(self.activation_func(np.vdot(W,x_i)),Y[i])
        return sum/n
    def fit(self,X,Y):
        
        Y = self.activation_func(Y)
        n_simples,_ = X.shape
        self.weight_s = self.w0
        self.weights_t = self.w0
        for t in range(self.T_max):
            # calculate w(t) using PLA
            for i in range(n_simples):
                if self.sign(self.weights_t,X[i])*Y[i] < 0:
                    self.weights_t += Y[i]*X[i]
            # Evaluate Ls(w(t))
            Ls_wt = self.emperical_error(X,Y,self.weights_t)
            Ls_ws = self.emperical_error(X,Y,self.weight_s)
            self.loss.append([t,Ls_ws])
            if Ls_wt < Ls_ws :
                self.weight_s = self.weights_t
        self.weights = self.weight_s
        return self.weights
    def predict(self,X):
        '''
        X: array of test set of samples
        '''
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