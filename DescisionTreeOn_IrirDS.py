import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split 

data =  pd.read_csv('Iris.csv')
print("Dataset" data.head())

def splitdataset(balance_data):  
    
    X = balance_data.values[:, 0:4] 
    Y = balance_data.values[:, 5]  
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100) 
    return X, Y, X_train, X_test, y_train, y_test 


def train_using_gini(X_train, X_test, y_train): 
  
    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5) 
   
    clf_gini.fit(X_train, y_train) 
    return clf_gini

def prediction(X_test, clf_object): 
 
    y_pred = clf_object.predict(X_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred  


def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: ", 
        confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
      
    print("Report : ", 
    classification_report(y_test, y_pred)) 



X, Y, X_train, X_test, y_train, y_test = splitdataset(data) 
clf_gini = train_using_gini(X_train, X_test, y_train) 
     
print("Results Using Gini Index:") 
       
y_pred_gini = prediction(X_test, clf_gini) 
cal_accuracy(y_test, y_pred_gini) 
