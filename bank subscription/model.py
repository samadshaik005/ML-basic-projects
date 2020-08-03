import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score,f1_score

dataset = pd.read_csv('cleaned.csv')
X=dataset[['duration', 'pdays', 'poutcome', 'cons.price.idx', 'cons.conf.idx',
       'nr.employed']]
Y=dataset['output_binary_y']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
'''
model=LogisticRegression(C=0.1,solver='liblinear')
model.fit(X_train,y_train)
predictions2=model.predict(X_test)
predictions2
print(accuracy_score(y_test,predictions2),f1_score(y_test,predictions2))
'''




#decision tree
from sklearn.tree import DecisionTreeClassifier      #81,62
model = DecisionTreeClassifier(criterion='entropy',max_depth=5)
model.fit(X_train,y_train)
predictions = model.predict(X_test)
predictions
from sklearn.metrics import accuracy_score,f1_score,log_loss
print(accuracy_score(y_test,predictions),f1_score(y_test,predictions))

pickle.dump(model, open('bank1.pkl','wb'))
'''
#k-nearest-neighbors
from sklearn.neighbors import KNeighborsClassifier
k=7
model=KNeighborsClassifier(n_neighbors=k)
model.fit(X_train,y_train)
predictions3 = model.predict(X_test)
print(accuracy_score(y_test, predictions3))
print(f1_score(y_test,predictions3))         #80,60
'''