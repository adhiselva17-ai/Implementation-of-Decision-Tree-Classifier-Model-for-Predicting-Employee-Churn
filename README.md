# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and preprocess the dataset by encoding categorical values and separating features and target variable.
2.Split the data, train a Decision Tree classifier using entropy, and fit the model on training data.
3.Predict test results, calculate accuracy, and visualize the decision tree model. 
 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: ADHI SELVAKUMAR R
RegisterNumber: 25008421 
*/
import pandas as pd
data = pd.read_csv("Employee.csv")
print("data.head():")
print(data.head())
print("\ndata.info():")
print(data.info())

print("\nisnull() and sum():")
print(data.isnull().sum())
print("data value counts():")
print(data["left"].value_counts())
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
print("\nEncoding salary column:")
data["salary"] = le.fit_transform(data["salary"])
print(data.head())
print("\nSelecting Features:")
x = data[["satisfaction_level",
          "last_evaluation",
          "number_project",
          "average_montly_hours",
          "time_spend_company",
          "Work_accident",
          "promotion_last_5years",
          "salary"]]
print(x.head())
y = data["left"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=100
)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy", random_state=100)
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
print("\nAccuracy value:")
accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)
print("\nData Prediction:")
print(dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]]))
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 10))
plot_tree(dt,
          feature_names=x.columns,
          class_names=["Not Left", "Left"],
          filled=True)
plt.show()
```

## Output:
<img width="572" height="335" alt="image" src="https://github.com/user-attachments/assets/713f7339-1063-4dd5-95b7-c7e67ee7e265" />
<img width="801" height="501" alt="image" src="https://github.com/user-attachments/assets/9782914a-32b2-4db7-973b-5eb61dbf2a91" />

<img width="384" height="300" alt="image" src="https://github.com/user-attachments/assets/ea436aae-8f40-41cc-be0e-66bf04bfaf0e" />
<img width="208" height="191" alt="image" src="https://github.com/user-attachments/assets/1f3c9215-7288-4480-91cf-29e44a065cdb" />
<img width="192" height="83" alt="image" src="https://github.com/user-attachments/assets/01a46325-672d-4e59-90ab-49d7591d23cc" />
<img width="532" height="326" alt="image" src="https://github.com/user-attachments/assets/336948e3-3009-4ce0-bb6a-cfed335d9182" />
<img width="128" height="31" alt="image" src="https://github.com/user-attachments/assets/29adca90-0f84-4900-aa87-4219e4f9de72" />
<img width="1452" height="718" alt="image" src="https://github.com/user-attachments/assets/106e6342-fc3d-453b-941f-18a33b51ca3b" />




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
