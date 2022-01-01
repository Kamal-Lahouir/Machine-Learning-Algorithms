import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt 
from src.modules.classification.perceptron import Percepton
from pocket import Pocket
from adalin import Adalin



# create a non linear data
X,y = datasets.make_moons(150, noise=0.1, random_state=0)

# visulize it (close the first window of the visualization so start training the model)
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()


# add column of 1 to X
X_hstacked = np.hstack((X,np.ones((len(X),1))))

# split the data to train and test set
X_train, X_test, Y_train ,Y_test = train_test_split(X_hstacked,y , test_size=0.2,shuffle=True)

# initiate the model
model = Pocket([0.1,0.003,1],T_max=100)

# train the model
model.fit(X_train,Y_train)

# predict
y_pred = model.predict(X_test)

# calculate the accuracy
print("Accuracy: ",model.accuracy(y_pred,Y_test))

# plot the emperical error
model.draw_loss()
