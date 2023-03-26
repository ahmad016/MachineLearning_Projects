## Housing Dataset Exploration and Pre-processing
This project predicts the prices of houses in Boston using the data collected from https://archive.ics.uci.edu/ml/machine-learning-databases/housing/

The data is saved in this repository under `data.csv` and attribute information is stored under `housing.names`. The file, `data.csv`, has been slightly manipulated, so I can learn simple ways to clean data. For the feature `RM`, some values were manually removed.

The goal of this project is to figure out how data can be visualized to see trends and what models should be used to calculate prices for houses using those trends.

This is explained throughout `price-predictor.ipynb`, all information related to the product as well as the code is stored in there. A final script, `price-predictor.py` is also created which uses `results.joblib` created by `price-predictor.ipynb` to make predictions.

As this is one of my very first projects in data manipulation / ML, there's bound to be some errors or rookie mistakes and I'd appreciate some feedback. You can reach me at https://www.linkedin.com/in/about-muhammad-cheema/