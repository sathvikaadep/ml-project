import joblib
import numpy as np

# Load model
model = joblib.load("../models/model.pkl")

# Sample input (Iris features)
# [sepal length, sepal width, petal length, petal width]
sample = np.array([[5.1, 3.5, 1.4, 0.2]])

# Predict
prediction = model.predict(sample)

print("Predicted Class:", prediction)