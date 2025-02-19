from typing import Tuple
import numpy as np
from typing import Callable
from sklearn.model_selection import train_test_split

def binarize(labels: list[str]) -> np.array:
    """Binarize the labels.

    Binarize the labels such that "Chinstrap" is 1 and "Adelie" is 0.

    Args:
        labels (list[str]): The labels to binarize.
    
    Returns:
        np.array: The binarized labels.
    """
    return np.array([1 if s == "Chinstrap" else 0 for s in labels])

def split_data(X: np.array, y: np.array, test_size: float=0.2, 
                random_state: float = 42, shuffle: bool = True) -> Tuple[np.array, np.array, np.array, np.array]:
    """Split the data into training and testing sets.

    NOTE:
        Please use the train_test_split function from sklearn to split the data.
        Ensure your test_size is set to 0.2.
        Ensure your random_state is set to 42.
        Ensure shuffle is set to True.

    Args:
        X (np.array): The independent variables.
        y (np.array): The dependent variables.
        test_size (float): The proportion of the data to use for testing.
        random_state (int): The random seed to use for the split.
        shuffle (bool): Whether or not to shuffle the data before splitting.

    """
    res = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)
    return (np.array(res[0]), np.array(res[1]), np.array(res[2]), np.array(res[3]))

def standardize(X_train: np.array, X_test: np.array) -> Tuple[np.array, np.array]:
    """Standardize the training and testing data.

    Standardize the training and testing data using the mean and standard deviation of
    the training set.

    Recall that your samples are rows and your features are columns. Your goal is to
    standardize along the columns (features).
    
    NOTE: Ensure you use the mean and standard deviation of the training set for
    standardization of BOTH training and testing sets. This will accomplish what
    we talked about in lecture, where it is imperative to standardize them 
    separately. You should NOT use data from the testing set to standardize the
    training set, because it would be leaking information from the testing set.
    You should NOT use data from the testing set to standardize the testing set,
    because it would be like looking into the future and impairs the generalization
    of the model.

    Args:
        X_train (np.array): The training data.
        X_test (np.array): The testing data.

    Returns:
        Tuple[np.array, np.array]: The standardized training and testing data.
    """
    # num_features = np.shape(X_train)[1]
    mean = X_train.mean(axis=0)
    # print('mean: ', mean)
    sd = X_train.std(axis=0)
    # print('std: ', sd)
    f = lambda x: (x - mean) / sd

    return (np.array([f(x) for x in X_train]), np.array([f(x) for x in X_test]))

def euclidean_distance(x1: np.array, x2: np.array) -> float:
    """Calculate the Euclidean distance between two points x1 and x2.

    Args:
        x1 (np.array): The first point.
        x2 (np.array): The second point.
    
    Returns:
        float: The Euclidean distance between the two points.
    """
    # dist_sum = 0
    # for x1_i, x2_i in zip(x1, x2):
    #     dist_sum += (x2_i - x1_i) ** 2
    # return dist_sum ** 0.5
    return np.linalg.norm(x1 - x2)

def cosine_distance(x1: np.array, x2: np.array) -> float:
    """Calculate the cosine distance between two points x1 and x2.

    Args:
        x1 (np.array): The first point.
        x2 (np.array): The second point.

    Returns:
        float: The cosine distance between the two points.
    """
    return 1 - np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
    
def knn(x: np.array, y: np.array, 
        sample: np.array, distance_method: Callable, k: int) -> int:
    """Perform k-nearest neighbors classification.

    Args:
        X (np.array): The training data.
        y (np.array): The training labels.
        sample (np.array): The point you want to classify.
        distance_method (Callable): The distance metric to use. This MUST 
            accept two np.arrays and return a float.
        k (int): The number of neighbors to consider as equal votes.
    
    Returns:
        int: The label of the sample.
    """

    distances = [(distance_method(x_i, sample), y_i) for x_i, y_i in zip(x,y)]
    
    distances.sort(key=lambda d: d[0])
    values, counts = np.unique([y for dist, y in distances[:k]], return_counts=True)
    
    return values[np.argmax(counts)]

def linear_regression(X: np.array, y: np.array) -> np.array:
    """Perform linear regression using the normal equation.

    NOTE: It is important that you concatenate a column of ones to the independent
    variables X. This will effectively add a bias term to the model.

    Args:
        X (np.array): The independent variables.
        y (np.array): The dependent variables.
    
    Returns:
        np.array: The weights for the linear regression model
                (including the bias term)
    """

    # 1. Concatenate the bias term to X using np.hstack.
    ones_col = np.ones((np.shape(X)[0], 1))
    X = np.hstack((ones_col, X))    
    # 2. Calculate the weights using the normal equation.
    Xt = np.transpose(X)
    return np.linalg.inv(Xt @ X) @ Xt @ y

def linear_regression_predict(X: np.array, weights: np.array) -> np.array:
    """Predict the dependent variables using the weights and independent variables.

    NOTE: It is important that you concatenate a column of ones to the independent
    variables X. This will effectively add a bias term to the model.

    Args:
        X (np.array): The independent variables.
        weights (np.array): The weights of the linear regression model.
    
    Returns:
        np.array: The predicted dependent variables.
    """
    # 1. Concatenate the bias term to X using np.hstack.
    ones_col = np.ones((np.shape(X)[0], 1))
    X = np.hstack((ones_col, X))
    # 2. Calculate the predictions.
    return X @ weights    

def mean_squared_error(y_true: np.array, y_pred: np.array) -> float:
    """Calculate the mean squared error.

    You should use only numpy for this calculation.

    Args:
        y_true (np.array): The true values.
        y_pred (np.array): The predicted values.
    
    Returns:
        float: The mean squared error.
    """
    y_delta = y_true - y_pred
    return np.dot(y_delta, y_delta) / len(y_pred)

def sigmoid(z: np.array) -> np.array:
    """Calculate the sigmoid function.

    Args:
        z (np.array): The input to the sigmoid function.
    
    Returns:
        np.array: The output of the sigmoid function.
    """
    return 1 / (1 + np.exp(-z))

def logistic_regression_gradient_descent(X: np.array, y: np.array, 
                                         learning_rate: float = 0.01, 
                                         num_iterations: int = 5000) -> np.array:
    
    """Perform logistic regression using gradient descent.

    NOTE: It is important that you concatenate a column of ones to the independent
    variables before performing gradient descent. This will effectively add
    a bias term to the model. The hstack function from numpy will be useful.

    NOTE: The weights should be initialized to zeros. np.zeros will be useful.

    NOTE: Please follow the formula provided in lecture to update the weights.
    Other algorithms will work, but the tests are expecting the weights to be
    calculated in the way described in our lecture.

    NOTE: The tests expect a learning rate of 0.01 and 5000 iterations. Do
    not change these values prior to submission.

    NOTE: This function expects you to use the sigmoid function you implemented
    above.

    Args:
        X (np.array): The independent variables.
        y (np.array): The dependent variables.
        learning_rate (float): The learning rate.
        num_iterations (int): The number of iterations to perform.
    
    Returns:
        np.array: The weights for the logistic regression model.
    """
    # 1. Concatenate the bias term to X using np.hstack.
    ones_col = np.ones((np.shape(X)[0], 1))
    X = np.hstack((ones_col, X))
    Xt = np.transpose(X)
    m = np.shape(X)[0]

    # 2. Initialize the weights with zeros. np.zeros is your friend here! 
    weights = np.zeros(X.shape[1])

    # For each iteration, update the weights.
    for _ in range(num_iterations):
        # 3. Calculate the predictions.
        y_pred = sigmoid(X @ weights)
        
        # 4. Calculate the gradient.
        J_gradient = (1 / m) * Xt @ (y_pred - y)

        # 5. Update the weights -- make sure to use the learning rate!
        # print("roc: ", learning_rate * J_gradient)
        # print('weights: ', weights)
        weights = weights - (learning_rate * J_gradient)

    # print(weights)
    return weights

def logistic_regression_predict(X: np.array, weights: np.array) -> np.array:
    """Predict the labels for the logistic regression model.

    NOTE: This function expects you to use the sigmoid function you implemented
    above.

    Args:
        X (np.array): The independent variables.
        weights (np.array): The weights of the logistic regression model. This
            should include the bias term.
    
    Returns:
        np.array: The output of logistic regression.
    """
    # 1. Add the bias term using np.hstack.
    ones_col = np.ones((np.shape(X)[0], 1))
    X = np.hstack((ones_col, X))

    # 2. Calculate the predictions using the provided weights.
    return sigmoid(X @ weights)
