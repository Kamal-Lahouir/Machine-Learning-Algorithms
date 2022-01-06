import numpy as np
import os
import sys
from sklearn import datasets
from sklearn.model_selection import train_test_split

#Path settings
sys.path.insert(0, os.path.abspath('../'))
# We import the model
from models_tobesaved.perceptron_cpy import Perceptron

# Create a non linear data
X,y = datasets.make_blobs(100, centers = 2 , random_state=5)

# We add to the X a row of ones because we are using a vectorized form of the weight and bias
X_hstacked = np.hstack((X,np.ones((len(X),1))))

# split the data to train and test set
X_train, X_test, Y_train ,Y_test = train_test_split(X_hstacked,y , test_size=0.2,shuffle=True)

# initiate the model
model = Perceptron()

# train the model
model.fit(X_train,Y_train)

# predict
y_pred = model.predict(X_test)

# calculate the accuracy
print("Accuracy: ",model.accuracy(y_pred,Y_test))

# plot the emperical error
model.draw_loss()

