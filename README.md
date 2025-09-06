# Tata Steel Stock High Price Prediction

## Overview
This project implements a **machine learning** model (Random Forest), served via FastAPI, to predict the "High" price for Tata Steel Ltd. stock given the "Open" and "Low" prices. The web app provides both a user interface and REST API for predictions.

**Live Demo:** [https://stockprediction-qott.onrender.com](https://stockprediction-qott.onrender.com)

## Table of Contents
- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Model Training](#model-training)
- [Model Performance](#model-performance)
- [Deployment](#deployment)
- [Technologies Used](#technologies-used)
- [Why Y Scaling Was Used](#why-y-scaling-was-used)
- [License](#license)

## Project Description

- **Machine Learning Model:** The project's notebook (`tisc.ipynb`) details the data preparation and training of a RandomForestRegressor to predict the "High" price using "Open" and "Low" as features.
- **API Backend:** `app.py` is a FastAPI app that loads the trained model (`randomforest_model.pkl`) and scaler (`robust_scaler.pkl`), and provides prediction functionality via web forms and API.
- **Frontend:** HTML templates in the `templates/` folder (`index.html`, `page2.html`) provide a clean user interface. Static files (CSS, images) are served from `static/`.

## Installation

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/hrishabh-dev/stockpredtatasteel
    cd stockpredtatasteel
    ```

2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(A Python virtual environment is recommended.)*

## Usage

1. **Start the FastAPI Server:**
    ```bash
    uvicorn app:app --reload
    ```
2. **Access the Application:**
    - Visit [http://localhost:8000](http://localhost:8000) for the web UI.
    - API docs available at [http://localhost:8000/docs](http://localhost:8000/docs).

## API Endpoints

- `GET /` — Home page (web form).
- `GET /predict` — Prediction form.
- `POST /predict` — Submit "Open" and "Low" prices, receive predicted "High" price as a response.

## Model Training

- Data is sourced from a Tata Steel historical dataset.
- Features: "Open", "Low". Target: "High".
- Processing in `tisc.ipynb`:
    - Cleans and preprocesses data.
    - Trains a RandomForestRegressor with RobustScaler preprocessing.
    - Evaluates on test data and saves both the model and scaler.

## Model Performance

- **Mean Squared Error (MSE):** ~0.51
- **Mean Absolute Error (MAE):** ~0.41
- **R² Score:** ~0.99

*(Metrics are from notebook evaluation; see `tisc.ipynb` for details.)*

## Deployment

- Hosted on Render at [https://stockprediction-qott.onrender.com](https://stockprediction-qott.onrender.com)
- To deploy elsewhere, connect your GitHub repo to a cloud platform and use:
    ```bash
    uvicorn app:app --host 0.0.0.0 --port 10000
    ```

## Technologies Used

- Python 3.x
- FastAPI
- Uvicorn
- Pandas, NumPy, scikit-learn
- Jinja2 (templating)
- HTML/CSS (frontend)
- Render (deployment)

## Why Y Scaling Was Used

**Y scaling** (scaling the target variable) was used in this project to improve the performance and stability of the Random Forest regression model. In stock price prediction, the target ("High" price) can have a wide range or outliers. Scaling ensures that the model treats all target values uniformly, helps in faster convergence, and prevents bias due to large target values. It also helps when the model is served for real-time predictions, ensuring the output is within a sensible, normalized range, and makes post-processing (inverse transforming to original values) straightforward for accurate user results.

---

## License

This project is licensed under the GNU General Public License (GPL) v3.0. See the [LICENSE](LICENSE) file for details.

---

*For questions or contributions, please use the GitHub issue tracker.*
