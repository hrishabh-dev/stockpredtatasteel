# Tata Steel Stock Price Prediction

Welcome to the Tata Steel Stock Price Prediction project! This repository is dedicated to building, training, and evaluating machine learning models that predict Tata Steel's stock price movements using historical data and neural networks.

---

## Project Overview

This project leverages deep learning (TensorFlow/Keras) and robust data pre-processing to predict the **High** price of Tata Steel's stock for a given day, using features such as **Open** and **Low** prices. The workflow is fully demonstrated in the included Jupyter notebook [`tisc.ipynb`](https://github.com/hrishabh-dev/stockpredtatasteel/blob/main/tisc.ipynb).

### Key Features

- **Historical Data Analysis**: Loads and visualizes 5,000 days of Tata Steel stock prices.
- **Data Preprocessing**: Cleans the data, drops unused columns, and applies robust scaling.
- **Feature Engineering**: Uses `Open` and `Low` prices as input features to predict the `High` price.
- **Model Building**: Builds a multilayer neural network with dropout for regularization.
- **Training & Evaluation**: Trains the model, reports metrics (MSE, MAE, R² score), and checks for overfitting.
- **Model Saving**: Saves trained models and scalers for future inference.

---

## Jupyter Notebook Workflow

See [`tisc.ipynb`](https://github.com/hrishabh-dev/stockpredtatasteel/blob/main/tisc.ipynb) for the full pipeline:

1. **Load Data**  
   Reads Tata Steel historical data CSV as a DataFrame.

2. **Preprocessing**  
   Drops unnecessary columns (`Date`, `Price`, `Vol.`, `Change %`).  
   Scales features (Open, Low) and the target (High) using `RobustScaler`.

3. **Train/Test Split**  
   Splits scaled data into training and testing sets (80/20).

4. **Model Architecture**  
   - 3 Dense layers: 256, 128, 64 neurons (ReLU activation)
   - Dropout layer (rate=0.1) added after each hidden layer for regularization
   - Final Dense layer for regression output

5. **Training**  
   - Optimizer: Adam (lr=0.001)
   - Loss: Mean Squared Error (MSE)
   - Epochs: 150, Batch size: 32

6. **Evaluation**  
   - Reports training and testing loss
   - Calculates R² score, MSE, and MAE on test data

7. **Model Export**  
   - Saves trained model and scalers using `joblib` and Keras save utilities

---

## Example Results

- **R² Score:** ~0.99 (indicating very high prediction accuracy on test data)
- **Test MSE:** ~0.0035
- **Test MAE:** ~0.048

---

## How to Run

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Install dependencies:  
  ```
  pip install pandas numpy scikit-learn tensorflow joblib
  ```

### Steps

1. Place your Tata Steel historical data CSV in the working directory (see notebook for expected format).
2. Open and run [`tisc.ipynb`](https://github.com/hrishabh-dev/stockpredtatasteel/blob/main/tisc.ipynb) in Jupyter or Google Colab.
3. Follow the workflow: load data, preprocess, train, evaluate, and save the model.

---

## Repository Structure

```
stockpredtatasteel/
├── tisc.ipynb                # Main Jupyter notebook for workflow
├── TISC Historical Data (1).csv # Historical stock data (not included)
├── scaler_X_standard.pkl     # Saved feature scaler
├── scaler_y_standard.pkl     # Saved target scaler
├── model.h5                  # Trained Keras model (after running notebook)
├── README.md                 # Project documentation (this file)
├── venv/                     # Optional: Python virtual environment
```

---

## Contributing

Feel free to open issues or pull requests to suggest improvements, report bugs, or add new features!

---

## License

This project is licensed under the MIT License.

---

## Author

Created by [hrishabh-dev](https://github.com/hrishabh-dev)

---
