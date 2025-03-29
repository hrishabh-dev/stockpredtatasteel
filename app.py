from flask import Flask,render_template,request,jsonify
import pandas as pd 
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler 
import pickle
import warnings
warnings.filterwarnings('ignore')
app=Flask(__name__) 
with open('x_scalerbest.pkl', 'rb') as f:
    X_scaler = pickle.load(f)

with open('y_scalerbest.pkl', 'rb') as f:
    y_scaler = pickle.load(f)

loaded_model = keras.models.load_model('mytatasteelbest.keras')


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=["GET","POST"])
def predict():        
    if request.method == "POST":
        try:
            open_price = request.form.get('open_price')
            low_price = request.form.get('low_price')
            open_price = float(open_price)
            low_price = float(low_price)

            # Input validation
            if open_price < 0 or low_price < 0:
                raise ValueError("Prices must be positive.")
            if open_price <= low_price:
                raise ValueError("Open price must be greater than low price.")
            
            # Prepare features for prediction
            float_features = [open_price, low_price]
            
            features = [np.array(float_features)]
            input_scaled = X_scaler.transform(features)  
            predicted_high_scaled = loaded_model.predict(input_scaled)  
            predicted_high = y_scaler.inverse_transform(predicted_high_scaled.reshape(-1, 1))
            prd = predicted_high[0][0]  
            return render_template("page2.html", prediction_text=f"Expected High Price is {prd:.2f}")
        except Exception as e:
            print("Error during prediction:", e)  # Catch the error and print
            return render_template("page2.html", prediction_text=f"An error occurred: {str(e)}")
    
    # Ensure there's always a return statement for GET requests as well
    return render_template("page2.html")  

if __name__=="__main__":
    app.run(debug=True)