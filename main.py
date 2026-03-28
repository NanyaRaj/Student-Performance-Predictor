import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
data = {
    'Study_Hours': [2, 3, 4, 5, 6, 7, 8, 9],
    'Sleep_Hours': [5, 6, 6, 7, 7, 8, 8, 9],
    'Attendance': [60, 65, 70, 75, 80, 85, 90, 95],
    'Marks': [50, 55, 60, 65, 70, 75, 85, 90]
    }
df = pd.DataFrame(data)
X = df[['Study_Hours', 'Sleep_Hours', 'Attendance']]
y = df['Marks']
model = LinearRegression()
model.fit(X, y)
study = float(input("Enter Study Hours: "))
sleep = float(input("Enter Sleep Hours: "))
attendance = float(input("Enter Attendance (%): "))
prediction = model.predict([[study, sleep, attendance]])
print("Predicted Marks:", prediction[0])
plt.scatter(df['Study_Hours'], df['Marks'])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.show()