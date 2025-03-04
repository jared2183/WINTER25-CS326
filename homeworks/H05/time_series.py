from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd

np.random.seed(2024)

def mse(predictions: np.ndarray, test: np.ndarray) -> float:
    """Mean Squared Error.

    NOTE: You may use only numpy. Do not use any other libraries.

    Args:
        predictions (np.ndarray): Predictions.
        test (np.ndarray): True values.

    Returns:
        float: Mean Squared Error.
    """
    return np.mean((test - predictions) ** 2)


def walk_forward_validation_arima(train: np.ndarray, test: np.ndarray, order: tuple) -> np.ndarray:
    """Walk-forward validation for ARIMA model.

    NOTE: You should use ARIMA model from statsmodels.

    NOTE: Please follow the provided steps in the function.

    Args:
        train (np.ndarray): Training data.
        test (np.ndarray): Test data.
        order (tuple): ARIMA order.

    Returns:
        np.ndarray: Predictions.
    """

    # Get current history.
    history = train.tolist()

    # Create a list to store predictions.
    predictions = []

    for sample in test:

        # 1. Initialize the ARIMA model from statsmodels.
        model = ARIMA(endog=history, order=order)

        # 2. Fit ARIMA model.
        res = model.fit()

        # 3. Forecast a single prediction (should be a float).
        pred = res.forecast()[0]

        # 4. Append prediction to predictions.
        predictions.append(pred)

        # 5. Append true value to history (i.e. walk-forward in loop).
        history.append(sample)

    # 6. Return np.ndarray of predictions.
    return np.array(predictions)


def local_seasonal_decompose(data: pd.DataFrame, model: str) -> DecomposeResult:
    """Seasonal decomposition.

    NOTE: data must have a datetime index with a frequency for this method to work.
    As long as you don't make any changes to the notebook, you should be fine.

    Args:
        data (pd.DataFrame): Input data.
        model (str): Model type (e.g. 'additive', 'multiplicative').

    Returns:
        DecomposeResult: Seasonal decomposition.
    """
    return seasonal_decompose(x=data, model=model)

def difference(data: np.ndarray, order: int) -> np.ndarray:
    """Difference the data.

    Args:
        data (np.ndarray): Input data.
        order (int): Order of differencing.
    
    Returns:
        np.ndarray: Differenced data.
    """
    return np.diff(data, n=order)


def is_stationary(data: np.ndarray, alpha: float) -> bool:
    """Check if the data is stationary using Augmented Dickey-Fuller test.

    NOTE: This method should return True if the data is stationary, 
    and False otherwise.

    Args:
        data (np.ndarray): Input data.
        alpha (float): Significance level.

    Returns:
        bool: True if stationary, False otherwise.
    """
    return (adfuller(x=data)[1] < alpha)