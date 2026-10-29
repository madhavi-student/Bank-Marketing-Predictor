# Bank Marketing Predictor Using Machine Learning

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
print("Loading Bank Marketing Dataset...")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"

import urllib.request
import zipfile
import io

response = urllib.request.urlopen(url)
zip_file = zipfile.ZipFile(io.BytesIO(response.read()))

with zip_file.open("bank-full.csv") as file:
    df = pd.read_csv(file, sep=";")

print("Dataset Loaded Successfully!")
print(df.head())

# Convert categorical data into numerical data
for column in df.select_dtypes(include="object").columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])

# Separate features and target
X = df.drop("y", axis=1)
y = df["y"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Training Completed!")
print("Model Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Predict a customer
sample = X_test.iloc[[0]]
prediction = model.predict(sample)

print("\nCustomer Prediction:")

if prediction[0] == 1:
    print("Customer may subscribe to a term deposit.")
else:
    print("Customer may not subscribe to a term deposit.")

# Save predictions
results = X_test.copy()
results["Prediction"] = y_pred

results.to_csv("bank_predictions.csv", index=False)

print("\nPredictions saved successfully!")
print("Bank Marketing Predictor completed!")