from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame(pd.read_csv("data_banknote_authentication.csv"))

x = df[['variance of image', 'skewness', 'kurtosis', "entropy"]]
y = np.array(df["fake"])
y.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

logRegrModel = LogisticRegression()
logRegrModel.fit(x_train, y_train)

y_predict = logRegrModel.predict(x_test)
conffussion = confusion_matrix(y_test, y_predict)
print(conffussion)
