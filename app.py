from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from fastapi.staticfiles import StaticFiles
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import pickle
import warnings

warnings.filterwarnings('ignore')

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

with open('x_scalerbest.pkl', 'rb') as f:
    X_scaler = pickle.load(f)

with open('y_scalerbest.pkl', 'rb') as f:
    y_scaler = pickle.load(f)

loaded_model = keras.models.load_model('mytatasteelbest.keras')


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
@app.get("/predict", response_class=HTMLResponse)  # Add this route to handle GET requests to /predict
async def show_prediction_form(request: Request):
    return templates.TemplateResponse("page2.html", {"request": request})
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, open_price: float = Form(...), low_price: float = Form(...)):
    try:
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
        return templates.TemplateResponse("page2.html", {"request": request, "prediction_text": f"Expected High Price is {prd:.2f}"})
    except Exception as e:
        print("Error during prediction:", e)  # Catch the error and print
        return templates.TemplateResponse("page2.html", {"request": request, "prediction_text": f"An error occurred: {str(e)}"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
