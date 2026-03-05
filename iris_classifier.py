import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# load dataset
iris = load_iris()

X = iris.data
y = iris.target

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# create model
model = LogisticRegression(max_iter=200)

# train model
model.fit(X_train, y_train)

# make prediction
prediction = model.predict([X_test[0]])

# convert number to flower name
flower_name = iris.target_names[prediction][0]

print("Predicted Flower:", flower_name)