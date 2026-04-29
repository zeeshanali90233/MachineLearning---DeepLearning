# Logistic Regression - Diabetes Prediction

This project demonstrates how to use **Logistic Regression** to predict whether a patient is likely to have diabetes based on medical measurements.

---

## What is Logistic Regression?

Logistic Regression is a supervised machine learning algorithm used for **classification** problems.

It estimates the probability of a class label such as `0` or `1` by applying the **sigmoid function**.

- `0` -> No diabetes
- `1` -> Diabetes

Although the name includes "regression", it is mainly used for classification tasks.

---

## Step 1: Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Step 2: Import Required Modules

The notebook uses:

- `pandas` for loading and handling the dataset
- `matplotlib` for plotting the confusion matrix
- `scikit-learn` for splitting data, training the model, and evaluating results

---

## Step 3: Load Dataset

The dataset is loaded from:

```python
df = pd.read_csv("diabetes.csv")
```

This dataset contains medical features and the target column `Outcome`.

---

## Step 4: Preview and Inspect Data

The notebook checks:

- first few rows using `df.head()`
- dataset shape using `df.shape`

This helps verify the dataset structure before training.

---

## Step 5: Define Features and Target

The model uses all columns except `Outcome` as input features.

```python
X = df.drop(["Outcome"], axis=1)
y = df["Outcome"]
```

---

## Step 6: Split the Dataset

The data is split into training and testing sets using the 80/20 rule.

```python
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42
)
```

---

## Step 7: Train the Logistic Regression Model

The notebook trains the model with:

```python
model = LogisticRegression(C=0.5, max_iter=2000)
model.fit(X_train, y_train)
```

### Parameters Used

- `C`: inverse of regularization strength
- Smaller `C` -> stronger regularization
- Larger `C` -> weaker regularization
- `max_iter`: maximum number of iterations for optimization

The higher `max_iter` value helps the model converge properly.

---

## Step 8: Make Predictions

After training, predictions are made on the test data.

```python
y_pred = model.predict(X_test)
```

---

## Step 9: Evaluate the Model

The notebook evaluates performance using:

- Accuracy score
- Confusion matrix
- Classification report
- 5-fold cross-validation accuracy

```python
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

Cross-validation is used to measure how stable the model is across different splits of the data.

---

## Step 10: Visualize the Confusion Matrix

The confusion matrix is plotted using `matplotlib` to compare predicted labels with true labels.

Labels used in the notebook:

- `No Diabetes`
- `Diabetes`

This makes model performance easier to interpret visually.

---

## Step 11: Test on New Sample Data

The notebook also predicts the class for a custom sample:

```python
data = [[0, 100, 80, 35, 0, 25.6, 0.201, 30]]
prediction = model.predict(data)
print("Prediction for data", prediction)
```

This shows how the trained model can be used on unseen input values.

---

## Conclusion

- Logistic Regression is a simple and effective classification algorithm
- It works well for binary prediction tasks like diabetes detection
- Model quality should be judged using both accuracy and class-wise metrics
- Cross-validation provides a better estimate of general performance