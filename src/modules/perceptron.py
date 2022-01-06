import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.001, n_iters=100):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.emperical_error = self.emperical_error
        self.activation_func = self._unit_step_func
        self.tf_func = self.truefalse_func
        self.weights = None
        self.bias = None
        self.loss = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.random.rand(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])
        
        for _ in range(self.n_iters):
            sum_error = 0.0

            for idx, x_i in enumerate(X):    
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)
                error = (y_predicted - y_[idx])
                sum_error += error**2
                # Perceptron update rule
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update
            Ls = sum_error / X.shape[0]
            print([_,Ls])
            self.loss.append([_,Ls])
            
            
            

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def emperical_error(self,X,y):
        
        # X : array of all simples where the rows are the simples
        # Y : array of predictions
        n = len(X)
        sum = 0
        for i,x_i in enumerate(X) :
            sum += self.tf_func(self.activation_func(np.vdot(self.weights,x_i)),y[i])
        return sum/n
    
    def truefalse_func(self,a,b):
        return 1 if a!=b else 0

    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, 0)

    # Loss drawing function: it gives the shape of the loss function through the list self.loss
    def draw_loss(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        X = np.array(self.loss)[:,0]
        Y = np.array(self.loss)[:,1]
        plt.ylabel("Emperical Error")
        plt.xlabel("Iterations")
        ymin = np.amin(0)
        ymax = np.amax(1)
        ax.set_ylim([ymin, ymax])

        plt.plot(X,Y)
        plt.show()
        plt.close('all')


