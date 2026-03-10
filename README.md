# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Load and preprocess the dataset by encoding categorical values and separating features and target variable.
Split the data, train a Decision Tree classifier using entropy, and fit the model on training data.
Predict test results, calculate accuracy, and visualize the decision tree model.. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: ADHI SELVAKUMAR R
RegisterNumber:  25008421
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
*/
```

## Output:

<img width="572" height="335" alt="image" src="https://github.com/user-attachments/assets/ed2d6e56-0991-4a88-a6da-7fb0ca5a1989" />
<img width="801" height="501" alt="image" src="https://github.com/user-attachments/assets/b6bd8880-b6c0-46db-b9d5-4d302deae630" />
<img width="384" height="300" alt="image" src="https://github.com/user-attachments/assets/979f465b-4968-437e-9759-89a38d3cb1df" />
<img width="208" height="191" alt="image" src="https://github.com/user-attachments/assets/4bc6aec8-072b-4646-879e-b889d0ce0d26" />
<img width="192" height="83" alt="image" src="https://github.com/user-attachments/assets/114b080a-23f1-4bab-a3ce-06a96db1144b" />
<img width="532" height="326" alt="image" src="https://github.com/user-attachments/assets/6ab3b47b-8f3a-4965-a44a-8ede3ff5b6ac" />
<img width="624" height="217" alt="image" src="https://github.com/user-attachments/assets/eb1d8fc8-e791-442c-9eff-b0c9b812a072" />

<img width="128" height="31" alt="image" src="https://github.com/user-attachments/assets/ff60a16b-4b44-40c8-8af7-1c94fd006571" />
<img width="1452" height="718" alt="image" src="https://github.com/user-attachments/assets/02a8cace-9e20-48a5-9a10-d3db854494b7" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
