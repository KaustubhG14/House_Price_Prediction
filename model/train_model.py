import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import joblib


df = pd.read_csv("dataset.csv")  


X = df[['bedrooms', 'bathrooms', 'sqft', 'location']]
y = df['price']


X['location'] = LabelEncoder().fit_transform(X['location'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


joblib.dump(model, "model/house_price_model.pkl")

print("Model trained and saved successfully.")
