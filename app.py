from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)


model = joblib.load("model/house_price_model.pkl")

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        
        bedrooms = int(request.form["bedrooms"])
        bathrooms = float(request.form["bathrooms"])
        sqft = int(request.form["sqft"])
        location = int(request.form["location"])  

       
        features = np.array([[bedrooms, bathrooms, sqft, location]])
        
       
        prediction = model.predict(features)[0]
        
        return render_template("index.html", prediction=round(prediction, 2))

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
