# Wind Power Generation Forecasting

This project focuses on forecasting wind power generation using machine learning techniques. It utilizes historical weather data and wind turbine specifications to predict future power output, aiding in better energy management and grid stability.

## Table of Contents

- [Wind Power Generation Forecasting](#wind-power-generation-forecasting)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Features](#features)
  - [Dataset](#dataset)
  - [Files in this Repository](#files-in-this-repository)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Model Training and Evaluation](#model-training-and-evaluation)
  - [Results](#results)
  - [Contributing](#contributing)
  - [License](#license)

## Project Overview

The core objective of this project is to develop a robust wind power generation forecasting model. Accurate wind power forecasts are crucial for grid operators to manage the intermittency of renewable energy sources, optimize dispatch schedules, and ensure reliable power supply. This repository contains the code for data preprocessing, model training, evaluation, and hyperparameter tuning.

## Features

* **Data Preprocessing:** Handles missing values, outliers, and scales features for model training.
* **Feature Engineering:** Potentially creates new features from existing ones to improve model performance (if applicable and implemented in the notebook).
* **Machine Learning Models:** Employs various machine learning algorithms for forecasting.
* **Hyperparameter Tuning:** Uses GridSearchCV to find the optimal hyperparameters for the chosen model.
* **Model Evaluation:** Evaluates model performance using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared ($R^2$).
* **Visualization:** Utilizes `matplotlib` and `seaborn` for data exploration and result visualization.

## Dataset

The project uses historical data from multiple wind farm locations. Each `LocationX.csv` file likely contains time-series data related to wind speed, direction, temperature, and corresponding power generation. The `merged_locations.csv` file is expected to be a consolidated dataset from all individual location files.

The key columns in the dataset are expected to include:

* **Time/Timestamp:** Date and time of the recording.
* **Wind Speed:** Wind speed at the turbine's hub height.
* **Wind Direction:** Direction from which the wind is blowing.
* **Temperature:** Ambient temperature.
* **Pressure:** Atmospheric pressure.
* **Humidity:** Relative humidity.
* **Power Output:** The actual wind power generated (target variable).
* Other relevant meteorological features.

## Files in this Repository

* `Wind_Power_Generation_Forecasting.ipynb`: The main Jupyter Notebook containing all the code for data loading, preprocessing, model building, training, evaluation, and hyperparameter tuning.
* `Location1.csv`: Dataset for Location 1.
* `Location2.csv`: Dataset for Location 2.
* `Location3.csv`: Dataset for Location 3.
* `Location4.csv`: Dataset for Location 4.
* `merged_locations.csv`: Combined dataset from all locations.

## Installation

To run this project, you need to have Python installed. It's recommended to use a virtual environment.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/wind-power-forecasting.git](https://github.com/yourusername/wind-power-forecasting.git)
    cd wind-power-forecasting
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    The notebook uses `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `seaborn`.
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
    Alternatively, you can find a `requirements.txt` file in a typical project. If not, the above command should cover the dependencies mentioned in the notebook.

## Usage

1.  **Open the Jupyter Notebook:**
    ```bash
    jupyter notebook Wind_Power_Generation_Forecasting.ipynb
    ```

2.  **Run the cells:** Execute each cell in the notebook sequentially. The notebook provides comments and markdown cells explaining each step.

    * **Data Loading:** The initial cells will load the `LocationX.csv` files and likely merge them into `merged_locations.csv`.
    * **Data Preprocessing:** Steps for cleaning and preparing the data for model training.
    * **Model Training:** The notebook will train the chosen machine learning model (e.g., RandomForestRegressor based on the snippet).
    * **Hyperparameter Tuning:** GridSearchCV will be used to optimize model parameters.
    * **Model Evaluation:** Performance metrics will be printed.

## Model Training and Evaluation

The `Wind_Power_Generation_Forecasting.ipynb` notebook details the model training process. It includes:

* **Data Splitting:** Dividing the dataset into training and testing sets.
* **Feature Scaling:** Applying `StandardScaler` to numerical features.
* **Model Selection:** The provided snippet indicates the use of `RandomForestRegressor`.
* **Hyperparameter Tuning:**
    python
    # Define the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': [0.6, 0.8, 1.0],
        'min_samples_leaf': [1, 5, 10]
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        scoring='neg_mean_absolute_error', # Optimize for MAE
        cv=3,  # 3-fold cross-validation
        n_jobs=-1  # Use all available CPU cores
    )

    # Fit the GridSearchCV
    grid_search.fit(X_train, y_train)

    # Best parameters and best score
    print(f'Best Parameters: {grid_search.best_params_}')
    print(f'Best MAE: {-grid_search.best_score_}')
    ```
* **Evaluation Metrics:**
    * Mean Absolute Error (MAE): $MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$
    * Mean Squared Error (MSE): $MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$
    * R-squared ($R^2$): $R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$

## Results

The notebook will output the performance metrics of the tuned model. For example:

...

## License

This project is open-sourced under the MIT License.
```
