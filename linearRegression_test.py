import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt 
import pandas as pd
from linearRegression import linearRegression


# create a non linear data
X,y = datasets.make_regression(n_samples=100,n_features=1,noise=20)
print(X.shape)
print(y.shape)
"""
df = pd.read_csv("data/cars.csv")

X = np.array(df['speed'])
X = X.reshape((X.shape[0],1))
print(X.shape)
y = np.array(df['dist'])
print(y.shape)
"""
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
model.draw_loss()

# plot of linear separator
y_pred_line = model.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, Y_train , color = cmap(0.9), s = 10)
m2 = plt.scatter(X_test, Y_test , color = cmap(0.5), s = 10)
plt.plot(X , y_pred_line, color = 'black', linewidth = 2 , label = "Prediction")
plt.show()

