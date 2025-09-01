from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import pandas as pd
import pickle
# Load your scaler and model at startup
@asynccontextmanager
async def lifespan(app:FastAPI):
    global X_scaler, loaded_model
    try:
        with open('robust_scaler.pkl', 'rb') as f:
            X_scaler = pickle.load(f)
    except Exception as e:
        print(f"Error loading X scaler: {e}")
        X_scaler = None

    try:
        with open('randomforest_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
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

        # Input validation
        if open_price < 0 or low_price < 0:
            raise ValueError("Prices must be positive.")
        if open_price <= low_price:
            raise ValueError("Open price must be greater than low price.")

        # Prepare input DataFrame
        input_df = pd.DataFrame([[open_price, low_price]], columns=['Open', 'Low'])

        # Scale input features
        if X_scaler is None:
            raise RuntimeError("Scaler not loaded.")
        input_scaled = X_scaler.transform(input_df)

        # Make prediction
        if loaded_model is None:
            raise RuntimeError("Model not loaded.")
        prediction_scaled = loaded_model.predict(input_scaled)

        # Since no y-scaling, prediction is direct
        predicted_high = prediction_scaled[0]

        return templates.TemplateResponse("page2.html", {"request": request, "prediction_text": f"Expected High Price is {predicted_high:.2f}"})
    except Exception as e:
        print("Error during prediction:", e)
        return templates.TemplateResponse("page2.html", {"request": request, "prediction_text": f"An error occurred: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
