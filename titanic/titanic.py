# %%

# Import necessary libraries
import numpy as np
import pandas as pd 

# Walk through the directory to print out all file paths within the directory
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

for dirname, _, filenames in os.walk(script_dir + '/data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %%

# Read data/train.csv and display the first 5 rows
train_data = pd.read_csv("data/train.csv")
train_data.head()

#%%

# Read data/test.csv and display the first 5 rows
test_data = pd.read_csv("data/test.csv")
test_data.head()

#%%

# Calculate the percentage of women who survived using tain_data (train.csv)
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("\n(train.csv) % of women who survived:", rate_women)

#%%

# Calculate the percentage of men who survived using tain_data (train.csv)
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("(train.csv) % of men who survived:", rate_men)

# %%

# Prepare the data for machine learning by creating features and target variables
from sklearn.ensemble import RandomForestClassifier

# Target variable
y = train_data["Survived"]

# Features to be used in the model
features = ["Pclass", "Sex", "SibSp", "Parch"]

# Create dummy variables for categorical variables
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# Initialize the Random Forest Classifier with given hyperparameters and train the model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)

# Make predictions on the test set (test.csv)
predictions = model.predict(X_test)

# Create a dataframe with passenger ID and predictions and save it as a CSV file
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)

# Let the user know if submission.csv was successfully created
print("\nYour submission was successfully saved!")

# %%
# Read data/test.csv and display the first 5 rows
submission_data = pd.read_csv("submission.csv")
submission_data.head()

#%%

# Calculate the percentage of women who survived using tain_data (train.csv)
train_total_survivors = train_data["Survived"].sum()
train_total_passengers = len(train_data)

train_survival_rate = (train_total_survivors / train_total_passengers)

print("\n% of individuals who survived from data provided in train.csv:", train_survival_rate)

#%%

# Calculate the percentage of men who survived using tain_data (train.csv)
submission_total_survivors = submission_data["Survived"].sum()
submission_total_passengers = len(submission_data)

submission_survival_rate = (submission_total_survivors / submission_total_passengers)

print("% of individuals who survived from data provided in submission.csv:", submission_survival_rate)

# %%

compare_survival_rate = abs(train_survival_rate - submission_survival_rate)

print(f"""
train.csv shows the survival rate to be {round(train_survival_rate * 100,2)}% 
submission.csv shows the survival rate to be {round(submission_survival_rate * 100,2)}%
The difference between the two is {round(compare_survival_rate * 100,2)}%""")


# %%
