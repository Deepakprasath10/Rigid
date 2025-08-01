import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

df = pd.read_csv('Used_car_data.csv')

X = df.drop(['Selling_Price', 'Name'], axis=1)
y = df['Selling_Price']


categorical = ['Fuel_Type', 'Seller_Type', 'Transmission']
numerical = ['Year', 'Present_Price', 'Kms_Driven', 'Owner']


preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical)
], remainder='passthrough')

preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(drop='first'), categorical)
], remainder='passthrough')

model = Pipeline([
    ('preprocessing', preprocessor),
    ('regressor', Ridge(alpha=1.0))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)


with open('ridge_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as ridge_model.pkl")
