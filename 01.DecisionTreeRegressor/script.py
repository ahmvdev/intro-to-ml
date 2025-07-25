import pandas as pd 
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

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

train_X, val_X,train_y, val_y = train_test_split(X, y)

melb_model = model.fit(train_X, train_y)

predictions = melb_model.predict(val_X)

print(predictions[:5])
print(val_y)


