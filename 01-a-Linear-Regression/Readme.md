# 🏠 Linear Regression - House Price Prediction

This project demonstrates how to use **Linear Regression** to predict house prices based on different features like area, bedrooms, bathrooms, etc.

---

## 📌 What is Linear Regression?

Linear Regression is a statistical method used to model the relationship between variables by fitting a straight line.

y = mx + c

- **y** → Dependent variable (Price)  
- **x** → Independent variable (Features like area)  
- **m** → Slope  
- **c** → Intercept  

---

## ⚙️ Step 1: Install Required Libraries

pip install pandas scikit-learn matplotlib seaborn

---

## 📥 Step 2: Import Libraries

import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt  

---

## 📊 Step 3: Load Dataset

df = pd.read_csv("Housing.csv")  

---

## 📈 Step 4: Data Visualization

- Price distribution using histogram  
- Area vs price using scatter plot  

---

## 🔄 Step 5: Data Preprocessing (Label Encoding)

Convert categorical values into numeric using LabelEncoder.

---

## ❗ Step 6: Check Missing Values

No missing values found.

---

## 🔗 Step 7: Feature Correlation

Area, bathrooms, and air conditioning strongly affect price.

---

## 🎯 Step 8: Define Features & Target

X = df.drop(["price"], axis=1)  
y = df["price"]  

---

## ✂️ Step 9: Train-Test Split (80/20)

Split dataset using train_test_split.

---

## 🤖 Step 10: Train Linear Regression Model

Train model using LinearRegression().

---

## 🔮 Step 11: Make Predictions

Predict using model.predict().

---

## 📏 Step 12: Evaluate Model

Metrics used:
- MAE
- MSE
- RMSE
- R² Score (~65% accuracy)

---

## 📉 Step 13: Visualize Predictions

Scatter plot of actual vs predicted values.

---

## 💾 Step 14: Save Model

Save model using pickle.

---

## 🔁 Step 15: Load Model & Predict

Load model and predict on new sample data.

---

## 🚀 Conclusion

- Linear Regression is simple and effective  
- Accuracy ~65%  
- Can improve with better features or advanced models  
