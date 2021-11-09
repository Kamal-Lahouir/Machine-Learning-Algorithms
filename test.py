import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt 
from linearRegression import linearRegression


# create a non linear data
X,y = datasets.make_regression(n_samples=100,n_features=1,noise=9)

# visulize it (close the first window of the visualization so start training the model)
plt.scatter(X,y)
plt.show()


# split the data to train and test set
X_train, X_test, Y_train ,Y_test = train_test_split(X,y , test_size=0.2,shuffle=True)

# initiate the model
model = linearRegression()

# train the model
model.fit(X_train,Y_train)

# predict
y_pred = model.predict(X_test)

# calculate the accuracy
print("Accuracy: ",model.accuracy(X_test,Y_test))

# plot the emperical error
# model.draw_loss()