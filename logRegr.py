from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame(pd.read_csv("data_banknote_authentication.csv"))

x = df[['variance of image', 'skewness', 'kurtosis']]
y = np.array(df["fake"])
y.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

logRegrModel = LogisticRegression()
logRegrModel.fit(x_train, y_train)

y_predict = logRegrModel.predict(x_test)
confusion = confusion_matrix(y_test, y_predict)

print("Welcome to the fake banknote detector! This is a logistic regression project I made for the ML/AI Engineer course from Codecademy.")

def menu():
    do=input("What do you want to do? \n1:take a look at how is the model working \n2:Add and test your own banknote\n3:take a look at the dataset used to train this model\n")
    if do == "1":
        try:
            print("True positives: ", confusion[0][0])
            print("True negatives: ", confusion[1][1])
            print("False positives: ", confusion[1][0])
            print("False negatives: ", confusion[0][1])
            menu()
        except:
            print("error. re-run the program")
    elif do == "2":
        try:
            var1 = input("input the skewness: ")
            var2 = input('variance of image: ')
            var3 = input("input the kurtosis: ")
            list = np.array([[float(var1),float( var2),float( var3)], [float(var1),float( var2),float( var3)]])
            pred = logRegrModel.predict(list)
            if pred[0] == 0:
                print("\nYour banknote is real\n\n")
                menu()
            else:
                print("\nYour banknote is fake\n\n")
                menu()
        except:
            print("error. re-run the program")
    elif do == "3":
        try:
            print(df.head(), "\n\n\n")
            print(df.info())
            menu()
        except:
            print("error. re-run the program")

menu()