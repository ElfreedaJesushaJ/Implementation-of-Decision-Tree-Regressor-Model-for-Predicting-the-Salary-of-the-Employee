# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import and preprocess the data
2. Split the dataset
3. Train the Decision Tree Regressor
4. Evaluate and predict

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Elfreeda Jesusha J
RegisterNumber:  212224040084
*/
```

```
import pandas as pd
data=pd.read_csv("/content/Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print(y_pred)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])

```

## Output:

data.head()

<img width="197" height="140" alt="ml9s1" src="https://github.com/user-attachments/assets/636328bc-991e-412d-a032-a58d3e1e13a8" />

data.info()

<img width="240" height="139" alt="ml9s2" src="https://github.com/user-attachments/assets/9a4537f6-220a-465e-b1ce-8093194cee89" />

data.isnull().sum()

<img width="123" height="124" alt="ml9s3" src="https://github.com/user-attachments/assets/437d216f-0db5-4863-bd49-fc1d964bb72f" />

data.head() after LabelEncoder

<img width="163" height="148" alt="ml9s4" src="https://github.com/user-attachments/assets/c14b84ea-e30e-42ef-97da-049c8f310178" />

y_pred Score

<img width="188" height="36" alt="ml9s5" src="https://github.com/user-attachments/assets/789ced4a-06cc-4b7e-85f8-c07cd7c60f67" />

Mean Square Error

<img width="162" height="36" alt="ml9s6" src="https://github.com/user-attachments/assets/0590f2b6-18c7-41c5-9a8c-1d2137e9b499" />

R2 Score

<img width="234" height="36" alt="ml9s7" src="https://github.com/user-attachments/assets/97c0a5ca-643d-4f3d-b944-682a33397b81" />

Prediction

<img width="715" height="36" alt="ml9s8" src="https://github.com/user-attachments/assets/9ce2961f-d6e3-47a4-91a2-d6c227109e18" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
