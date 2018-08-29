from sklearn.grid_search import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append( './src')
from plotter_functions import plot_confusion_matrix

data = pd.read_pickle("data/household_features")
ids = data.drop("idhogar", axis=1)
X = data.drop(["Target", "idhogar"], axis=1)
Y = data.Target.astype(int)

N = np.arange(len(X.index))
split = 0.2
n_train = int((1-split)*len(N))
n_test = len(N) - n_train

i_train = np.random.choice(N, n_train, replace=False)
i_test = np.array([i for i in N if i not in i_train])

x_train = X.values[i_train]
x_test = X.values[i_test]

y_train = Y[i_train]
y_test = Y[i_test]

RF = RandomForestClassifier()
RF.fit(x_train, y_train)
y_pred = RF.predict(x_test)

conf = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf, 
                        ["1", "2", "3", "4"],
                        True,
                        )
