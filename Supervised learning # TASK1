import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import urllib
import requests

dataset = pd.read_csv(urllib.request.urlopen("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"))
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


y_pred = regressor.predict(X_test)

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Marks vs Hours Of Study (Training set)')
plt.xlabel('Hours of Study')
plt.ylabel('Marks')
plt.show()

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Marks vs Hours Of Study (Testing set)')
plt.xlabel('Hours of Study')
plt.ylabel('Marks')
plt.show()
