# Titanic - Machine Learning from Disaster

## Goal
Use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.

---
Links:
- https://www.kaggle.com/competitions/titanic
- https://www.kaggle.com/code/alexisbcook/titanic-tutorial/notebook
---

### Prerequisites (Ubuntu / Debian)
1. Python3 and python3-pip: `sudo apt install python3 python3-pip`
2. Use pip to install required modules: (when in titanic directory) `pip install -r requirements.txt`

## Outcome
I used the data from test.csv and inputted it into a rainforest model with 100 trees (estimators). Once the model makes its predictions, they are saved in the form of a csv file called `submission.csv`. The file contains a PassengerId and whether they survived (`0` = **No** and `1` = **Yes**). 

Here are the results from the submission.csv that exists in this repository (you're free to delete it before executing the python script although the results will likely be the same/similar)

```
(train.csv) % of women who survived: 0.7420382165605095
(train.csv) % of men who survived: 0.18890814558058924

% of individuals who survived from data provided in train.csv: 0.3838383838383838
% of individuals who survived from data provided in submission.csv: 0.35406698564593303

train.csv shows the survival rate to be 38.38% 
submission.csv shows the survival rate to be 35.41%
The difference between the two is 2.98%
```