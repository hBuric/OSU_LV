import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn . metrics import accuracy_score, recall_score, precision_score
from sklearn . metrics import confusion_matrix, ConfusionMatrixDisplay


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

#a
plt.scatter(X_train[y_train == 0,0], X_train[y_train==0,1], color="blue")
plt.scatter(X_train[y_train == 1,0], X_train[y_train==1,1], color="red")
plt.scatter(X_test[y_test == 0,0], X_test[y_test==0,1], color="blue", marker='x')
plt.scatter(X_test[y_test == 1,0], X_test[y_test==1,1], color="red",  marker='x')
plt.show()

#b
LogRegression_model=LogisticRegression()
LogRegression_model.fit(X_train,y_train)

#c
theta0 = LogRegression_model.intercept_[0]
theta1, theta2 = LogRegression_model.coef_[0]
plt.scatter(X_train[y_train == 0,0], X_train[y_train==0,1], color="blue")
plt.scatter(X_train[y_train == 1,0], X_train[y_train==1,1], color="red")
plt.plot(X_train[:,0], (-theta0 - theta1 * X_train[:,0])/theta2)
plt.show()

#d
y_test_pred = LogRegression_model.predict(X_test)
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()
print("tocnost: " + str(accuracy_score(y_test, y_test_pred)))
print("odziv: " + str(recall_score(y_test, y_test_pred)))
print("preciznost: " + str(precision_score(y_test, y_test_pred)))

#e
plt.scatter(X_test[y_test == y_test_pred,0], X_test[y_test==y_test_pred,1], color="green")
plt.scatter(X_test[y_test != y_test_pred,0], X_test[y_test!=y_test_pred,1], color="black")
plt.show()
