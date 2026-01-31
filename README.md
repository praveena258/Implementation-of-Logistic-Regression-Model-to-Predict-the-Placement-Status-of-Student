# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: PRAVEENA D
RegisterNumber:  212224040248
*/

import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()
data1=data.copy()
data1.head()
data1=data1.drop(['sl_no','salary'],axis=1)
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:, : -1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy: ",accuracy)
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print(confusion)
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:


data.head:

<img width="880" height="163" alt="image" src="https://github.com/user-attachments/assets/08f768ce-fa7f-4771-b699-18eff0d06ee3" />


data1.head:

<img width="836" height="162" alt="image" src="https://github.com/user-attachments/assets/b24a968f-7dab-4f40-8d8e-ca9fa8bb4696" />


data1.isnull().sum():

<img width="190" height="234" alt="image" src="https://github.com/user-attachments/assets/44524fce-7116-4248-a96d-f3c25b8237c2" />


data1:

<img width="729" height="321" alt="image" src="https://github.com/user-attachments/assets/5bc46438-46a8-4cf9-8b9c-6a71a665aded" />


x:

<img width="694" height="332" alt="image" src="https://github.com/user-attachments/assets/0125698c-2112-4c4b-a36a-59cc2426b673" />



y:

<img width="508" height="212" alt="image" src="https://github.com/user-attachments/assets/72a592a8-78d7-4236-a861-c73dea1ef4f7" />


y_pred:

<img width="678" height="59" alt="image" src="https://github.com/user-attachments/assets/59fc0b51-d298-4672-9645-517d6b41aa3c" />


Accuracy:

<img width="192" height="45" alt="image" src="https://github.com/user-attachments/assets/2b2411d9-646f-47c1-b957-42a17b3a35b3" />


Confusion:

<img width="479" height="85" alt="image" src="https://github.com/user-attachments/assets/150d93e0-bc12-41b2-8269-76e5deeae0e8" />



Classification Report1:

<img width="1375" height="363" alt="image" src="https://github.com/user-attachments/assets/0adfc2f2-00ac-486f-b877-4e2739e1f1a1" />




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
