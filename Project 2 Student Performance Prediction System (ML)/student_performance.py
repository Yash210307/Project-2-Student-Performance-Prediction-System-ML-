# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
data = pd.read_csv("student_data.csv")

# Display Dataset
print("Dataset:")
print(data.head())

# Check Missing Values
print("\nMissing Values:")
print(data.isnull().sum())

# Dataset Information
print("\nDataset Info:")
print(data.info())

# -------------------------------
# Exploratory Data Analysis (EDA)
# -------------------------------

# Correlation Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(data.corr(), annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
plt.show()

# Attendance Distribution
plt.figure(figsize=(6,4))
sns.histplot(data["Attendance"], bins=10, kde=True)
plt.title("Attendance Distribution")
plt.show()

# Study Hours vs Performance
plt.figure(figsize=(6,4))
sns.scatterplot(x="StudyHours", y="FinalPerformance", data=data)
plt.title("Study Hours vs Final Performance")
plt.show()

# --------------------------------
# Feature Selection
# --------------------------------

X = data[['Attendance', 'StudyHours', 'Marks']]
y = data['FinalPerformance']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# --------------------------------
# Model Training
# --------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# --------------------------------
# Model Evaluation
# --------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("MAE :", mae)
print("MSE :", mse)
print("R2 Score :", r2)

# --------------------------------
# Predict New Student Performance
# --------------------------------

attendance = float(input("Enter Attendance (%): "))
study_hours = float(input("Enter Study Hours: "))
marks = float(input("Enter Marks: "))

new_student = np.array([[attendance, study_hours, marks]])

prediction = model.predict(new_student)

print(f"\nPredicted Final Performance: {prediction[0]:.2f}")