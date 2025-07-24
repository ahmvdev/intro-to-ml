import pandas as pd 
from sklearn.tree import DecisionTreeRegressor

melb_file_path = '../Kaggle-Datasets/melb_data.csv'

melb_data = pd.read_csv(melb_file_path)

melb_data = melb_data.dropna(axis=0)

melb_features = [
    'Rooms',
    'Bathroom',
    'Car',
    'Landsize',
    'BuildingArea',
    'YearBuilt',
    'Lattitude',
    'Longtitude'
]

X = melb_data[melb_features]

y = melb_data.Price

model = DecisionTreeRegressor(random_state=1)

melb_model = model.fit(X, y)

predictions = melb_model.predict(X)

print(y.head())
print(predictions)


