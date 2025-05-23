import numpy as np


def create_design_matrix_dataset_1(X_data: np.ndarray) -> np.ndarray:
    """
    Create the design matrix X for dataset 1.
    :param X_data: 2D numpy array with the data points
    :return: Design matrix X
    """
    # TODO: Create the design matrix X for dataset 1
    X = np.c_[X_data, X_data[:, 0] * X_data[:, 1], X_data[:, 0] ** 2, X_data[:, 1] ** 2, X_data[:, 0] ** 2 * X_data[:, 1] ** 2]

    assert X.shape[0] == X_data.shape[0], """The number of rows in the design matrix X should be the same as
                                             the number of data points."""
    assert X.shape[1] >= 2, "The design matrix X should have at least two columns (the original features)."
    return X


def create_design_matrix_dataset_2(X_data: np.ndarray) -> np.ndarray:
    """
    Create the design matrix X for dataset 2.
    :param X_data: 2D numpy array with the data points
    :return: Design matrix X
    """
    # TODO: Create the design matrix X for dataset 2
    X = np.c_[X_data, X_data[:, 0] ** 2, X_data[:, 1] ** 2]

    assert X.shape[0] == X_data.shape[0], """The number of rows in the design matrix X should be the same as
                                             the number of data points."""
    assert X.shape[1] >= 2, "The design matrix X should have at least two columns (the original features)."
    return X


def create_design_matrix_dataset_3(X_data: np.ndarray) -> np.ndarray:
    """
    Create the design matrix X for dataset 3.
    :param X_data: 2D numpy array with the data points
    :return: Design matrix X
    """
    # TODO: Create the design matrix X for dataset 3
    X = np.c_[X_data, X_data[:, 0] ** 2, X_data[:, 0] ** 3, X_data[:, 0] ** 4, X_data[:, 0] ** 5, X_data[:, 0] ** 6, X_data[:, 0] ** 7, X_data[:, 0] ** 8]
    
    assert X.shape[0] == X_data.shape[0], """The number of rows in the design matrix X should be the same as
                                             the number of data points."""
    assert X.shape[1] >= 2, "The design matrix X should have at least two columns (the original features)."
    return X


def logistic_regression_params_sklearn():
    """
    :return: Return a dictionary with the parameters to be used in the LogisticRegression model from sklearn.
    Read the docs at https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """
    # TODO: Try different `penalty` parameters for the LogisticRegression model
    return {'penalty': 'l2', 'solver': 'lbfgs'}
