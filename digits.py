import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


whole_data_set = 32818
train_data_set = 10

data = pd.read_csv("train.csv").as_matrix()
print(len(data))
clf = DecisionTreeClassifier()
xtrain = data[0:train_data_set,1:]
train_label=data[0:train_data_set,0]

clf.fit(xtrain, train_label)

xtest = data[train_data_set:,1:]
actual_label = data[train_data_set:,0]

# d= xtest[8]
# d.shape=(28,28)
# pt.imshow(255-d, cmap= "gray")
# print(clf.predict([xtest[8]]))
# pt.show()

p = clf.predict(xtest)

count = 0

for i in range(0,train_data_set):
    count +=1 if p[i]==actual_label[i] else 0
print(count)