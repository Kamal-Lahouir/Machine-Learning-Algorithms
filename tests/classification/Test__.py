from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from logisticRegression import logisticRegression
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

print(X.shape)
print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

regressor = logisticRegression(lr=0.0001, n_iters=1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
accuracy = regressor.accuracy(predictions,y_test)

# plot the emperical error
regressor.draw_loss()


print("LR classification accuracy:", accuracy)