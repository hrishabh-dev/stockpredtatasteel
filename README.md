# Deep Learning Model for Tata Steel Ltd

## Overview
This project implements a deep learning model to forecast performance metrics for Tata Steel Ltd. It utilizes FastAPI for serving the model and provides a user-friendly web interface built with HTML and CSS. 

You can view the project live at [My Website](https://stockprediction-qott.onrender.com).

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Deployment](#deployment)
- [Technologies Used](#technologies-used)
- [License](#license)
- [Demo](#demo)

## Description
In this project, I developed a deep learning model aimed at predicting key performance indicators for Tata Steel Ltd. Initially, I attempted to use Flask for deployment but encountered some issues, which led me to switch to FastAPI. The model achieves impressive performance metrics, with an accuracy of 99%.

## Installation
To run this project locally, please follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/hrishabh-dev/stockpredtatasteel
   cd stockpredtatasteel
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
After installing the necessary dependencies, you can start the FastAPI server with the following command:

```bash
uvicorn app:app --reload
```

## Model Performance
- **Accuracy:** 99%
- **Mean Squared Error (MSE):** 0.36
- **Mean Absolute Error (MAE):** 0.39

These results indicate that the model has high predictive accuracy and low error rates, showcasing its effectiveness for the problem at hand.

## Deployment
The web application is deployed using Render, which allows for seamless hosting of FastAPI applications. The backend server is run using Uvicorn, ensuring efficient handling of requests.

## Technologies Used
- Python
- FastAPI
- Uvicorn
- HTML/CSS
- Deep Learning (TensorFlow)
- Deployment with Render

## License
This project is licensed under the GNU General Public License (GPL) v3.0. See the [LICENSE](LICENSE) file for details.



