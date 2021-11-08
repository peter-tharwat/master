import pandas as pd
import pydotplus
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
import numpy as np


# 0 = Male
# 1 = FeMale

data = pd.DataFrame({'P_Movies': [17,64,18,20,38,49,55,25,29,31,33], 'Gender': [1,0,1,0,1,0,0,1,1,0,1]})
data =data.sort_values('P_Movies')
#data

#define Decision Tree
dt = DecisionTreeClassifier(criterion = 'entropy')
#Define input vectors
#X is the features in this dataset
X = data['P_Movies'].values.reshape(-1, 1)
#Y is the vector with our Target Variables
Y = data['Gender'].values
#start fitting process
dt.fit(X, Y)


d  = np.array([7, 15, 43, 45]).reshape(-1, 1)

#print(d)
print(dt.predict(d).reshape(-1, 1))