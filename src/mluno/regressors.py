import numpy as np

class KNNRegressor:
    """
    A class used to represent a K-Nearest Neighbors Regressor.
    Parameters
    ----------
    k : int
        The number of nearest neighbors to consider for regression.

    """
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        """
        Fit the model using X as input data and y as target values.
        Parameters
        ----------
        X : ndarray
            The training data, which is a 2D array of shape (n_samples, 1) where each row is a sample and each column is a feature.
        y : ndarray
            The target values, which is a 1D array of shape (n_samples, ).
        """
        self.X = X
        self.y = y

    def __repr__(self) -> str:
        return f"Knn Regression model with k = {self.k}."
    
    def predict(self, X_new):
        """
        Predict the target for the provided data.
        Parameters
        ----------
        X_new : ndarray
            Input data, a 2D array of shape (n_samples, 1), with which to make predictions.
        Returns
        -------
        ndarray
            The target values, which is a 1D array of shape (n_samples, ).
        """
        predicted_labels = [self._predict(x) for x in X_new]
        return np.array(predicted_labels)

    def _predict(self, x_new):
        distances = [np.linalg.norm(x - x_new) for x in self.X]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_y = self.y[k_indices]
        return np.mean(k_nearest_y)

class LinearRegressor:
    """
    A class used to represent a Simple Linear Regressor.
     $$ Y = \\beta_0 + \\beta_1 \\cdot x + \\epsilon $$


    Attributes
    ----------
    weights : ndarray
        The weights of the linear regression model. Here, the weights are represented by the $\\beta$ vector vector which for univariate regression is a 1D vector of length two, $\\beta = [\\beta_0, \\beta_1]$ where $\\beta_0$ is the slop and $\\beta_1$ is the intercept.
    """
    def __init__(self):
        self.beta = None

    def fit(self, X, y):
        """
        Trains the linear regression model using the given training data.
        In other words, the fit method learns the weights, represented by the $\\beta$ vector. To learn the $\\beta$ vector, use
        $${\\hat\\beta  = (X^TX)^{-1}X^Ty}$$

        Here, $X$ is the so-called design matrix, which, to include a term for the intercept, has a column of ones appended to the input X matrix.

        Parameters
        ----------
        X : ndarray
            The training data, which is a 2D array of shape (n_samples, 1) where each row is a sample and each column is a feature.
        y : ndarray
            The target values, which is a 1D array of shape (n_samples, ).

        """
        X = np.c_[np.ones(X.shape[0]), X]
        self.beta = np.linalg.inv(X.T @ X) @ X.T @ y

    def __repr__(self) -> str:
        return f"Linear Regression model with beta = {self.beta}."
    
    def predict(self, X_new):
        """
        Makes predictions for input data.
        $$ \\hat y = X\\hat\\beta $$
        Parameters
        ----------
        X_new : ndarray
            Input data, a 2D array of shape (n_samples, 1), with which to make predictions.
        Returns
        -------
        ndarray
            The predicted target values as a 1D array with the same length as X.

        """
        X_new = np.c_[np.ones(X_new.shape[0]), X_new]
        return X_new @ self.beta

# class BinnedRegressor:
#     def __init__(self, n_bins=10):
#         self.n_bins = n_bins

#     def fit(self, X, y):
#         self.bin_edges = np.linspace(X.min(), X.max(), self.n_bins + 1)
#         self.bin_centers = (self.bin_edges[1:] + self.bin_edges[:-1]) / 2
#         self.y = np.array([y[(X >= self.bin_edges[i]) & (X < self.bin_edges[i+1])].mean() for i in range(self.n_bins)])

#     def __repr__(self) -> str:
#         return f"Binned Regression model with n_bins = {self.n_bins}."
    
#     def predict(self, X_new):
#         return np.interp(X_new, self.bin_centers, self.y)