import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('datasets/Smarket.csv', usecols = range(1,10))

#Load train data (before 2005), and test data (in 2005)
x_train = data[data.Year < 2005][['Lag1', 'Lag2']]
y_train = data[data.Year < 2005]['Direction']
x_test = data[data.Year==2005][['Lag1', 'Lag2']]
y_test = data[data.Year==2005]['Direction']

#n = 1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)

y_predict = knn.predict(x_test)
print("Classification Report for k=1")
print(classification_report(y_test, y_predict, digits=3))
print(pd.DataFrame(confusion_matrix(y_test, y_predict).T,['Down', 'Up'], ['Down', 'Up']))

#n = 3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)

y_predict = knn.predict(x_test)
print("Classification Report for k=3")
print(classification_report(y_test, y_predict, digits=3))
print(pd.DataFrame(confusion_matrix(y_test, y_predict).T,['Down', 'Up'], ['Down', 'Up']))

#n = 5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)

y_predict = knn.predict(x_test)
print("Classification Report for k=3")
print(classification_report(y_test, y_predict, digits=3))
print(pd.DataFrame(confusion_matrix(y_test, y_predict).T,['Down', 'Up'], ['Down', 'Up']))
