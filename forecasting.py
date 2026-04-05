# ===============================================
# AI-Powered Energy Consumption Forecasting System
# ===============================================
# Author: Aman Tiwari
# Mentor: Umesh Yadav Sir
# Description: Predict energy consumption using Random Forest ML
# ===============================================

# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ===============================================
# 2️⃣ Create Sample Dataset (Synthetic Data)
# ===============================================
np.random.seed(42)
n = 500  # number of data points

data = pd.DataFrame({
    "temperature": np.random.uniform(15, 40, n),
    "humidity": np.random.uniform(30, 90, n),
    "hour": np.random.randint(0, 24, n),
    "day": np.random.randint(1, 31, n)
})

# Target variable: energy consumption
data["energy_consumption"] = (
    2*data["temperature"] +
    0.5*data["humidity"] +
    3*data["hour"] +
    np.random.normal(0, 10, n)
)

print("Sample Data:\n", data.head())

# ===============================================
# 3️⃣ Data Cleaning
# ===============================================
data = data.dropna()

# ===============================================
# 4️⃣ Feature Engineering
# ===============================================
data["temp_humidity_interaction"] = data["temperature"] * data["humidity"]

# ===============================================
# 5️⃣ Split Features and Target
# ===============================================
X = data.drop("energy_consumption", axis=1)
y = data["energy_consumption"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================================
# 6️⃣ Model Training (Random Forest)
# ===============================================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ===============================================
# 7️⃣ Prediction
# ===============================================
y_pred = model.predict(X_test)

# ===============================================
# 8️⃣ Evaluation
# ===============================================
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:\nRMSE: {rmse:.2f}\nR2 Score: {r2:.2f}")

# ===============================================
# 9️⃣ Visualization
# ===============================================
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linewidth=2)
plt.xlabel("Actual Energy Consumption")
plt.ylabel("Predicted Energy Consumption")
plt.title("Actual vs Predicted Energy Consumption")
plt.grid(True)
plt.show()

# ===============================================
#  🔹 Project Explanation 
# ===============================================
"""
Project Overview:
- Predicts energy consumption using environmental and time features
- Uses Random Forest Regressor
- Performs data preprocessing, feature engineering, model evaluation, and visualization

Why it is important:
- Helps industries and power companies forecast electricity usage
- Reduces cost and energy waste
- Applicable for smart grids, manufacturing, and buildings

Key Steps:
1. Create synthetic dataset (temperature, humidity, hour, day)
2. Feature engineering: interaction between temperature and humidity
3. Split dataset into train-test
4. Train Random Forest model
5. Predict energy consumption on test set
6. Evaluate using RMSE and R2 Score
7. Visualize actual vs predicted
"""