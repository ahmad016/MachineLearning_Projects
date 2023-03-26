from joblib import load
import numpy as np


model = load('results.joblib')

# Create an array of features to be predicted
features = np.array([[0.33147,0,6.2,0,0.507,8.247,70.4,3.6519,8,307,17.4,378.95,3.95]])


# Predict the MEDV value using the loaded model
target_value = model.predict(features)

# Print the predicted MEDV value
print("MEDV (Median value of owner-occupied homes in $1000's): ", target_value)
