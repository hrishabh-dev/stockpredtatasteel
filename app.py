from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import pandas as pd
import joblib # Use joblib for scalers
import tensorflow as tf # Import tensorflow
import numpy as np # Import numpy

# Load your scaler and model at startup
@asynccontextmanager
async def lifespan(app:FastAPI):
    global scaler_X, scaler_y, loaded_model
    try:
        # Load the correct scalers saved from the standard workflow
        scaler_X = joblib.load('saved_model/robust_X_standard.pkl')
        scaler_y = joblib.load('saved_model/robust_y_standard.pkl')
        print("Scalers loaded successfully.")
    except Exception as e:
        print(f"Error loading scalers: {e}")
        scaler_X = None
        scaler_y = None

    try:
        # Load the TensorFlow model saved from the standard workflow
        loaded_model = tf.keras.models.load_model('saved_model/stock_price_model_standard.keras')
        print("TensorFlow model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        loaded_model = None
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/predict", response_class=HTMLResponse)
async def show_prediction_form(request: Request):
    return templates.TemplateResponse("page2.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, open_price: float = Form(...), low_price: float = Form(...)):
    try:
        open_price = float(open_price)
        low_price = float(low_price)

        # Input validation (optional, but good practice)
        if open_price < 0 or low_price < 0:
             raise ValueError("Prices must be positive.")
        # This validation might be too strict depending on your data
        # if open_price <= low_price:
        #     raise ValueError("Open price must be greater than low price.")


        # Prepare input as a NumPy array (matching the format used during training)
        user_input = np.array([[open_price, low_price]])

        # Scale input features using the loaded scaler_X
        if scaler_X is None:
            raise RuntimeError("Scaler X not loaded.")
        input_scaled = scaler_X.transform(user_input)

        # Make prediction using the loaded TensorFlow model
        if loaded_model is None:
            raise RuntimeError("Model not loaded.")
        # TensorFlow models predict expects a batch, so input needs to be reshaped if it's a single sample
        # However, .predict handles single samples correctly if input shape matches model's input layer
        scaled_prediction = loaded_model.predict(input_scaled)

        # Inverse transform the scaled prediction using the loaded scaler_y
        if scaler_y is None:
             raise RuntimeError("Scaler Y not loaded.")
        # The scaler_y was fitted on the 'High' column (y), which is a single feature.
        # scaled_prediction is also a single value (the predicted scaled 'High').
        # Inverse transform expects input with the same number of features as the scaler was fitted on.
        # Since scaler_y was fitted on a single feature ('High'), scaled_prediction (which is a single value)
        # needs to be reshaped to a 2D array with shape (n_samples, 1) for inverse_transform.
        predicted_high_price = scaler_y.inverse_transform(scaled_prediction)


        # predicted_high_price will be a 2D array like [[predicted_value]].
        # Extract the single predicted value.
        predicted_high = predicted_high_price[0][0]


        return templates.TemplateResponse("page2.html", {"request": request, "prediction_text": f"Expected High Price is {predicted_high:.2f}"})
    except Exception as e:
        print("Error during prediction:", e)
        return templates.TemplateResponse("page2.html", {"request": request, "prediction_text": f"An error occurred: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
