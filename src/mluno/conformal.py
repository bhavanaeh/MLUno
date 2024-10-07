import numpy as np

class ConformalPredictor:
    """
    A class used to represent a (Split) Conformal Predictor.
    $$\\hat{C}\\_{alpha}(x) = \\left[ \\hat{u}(x) \\pm q_{1-\\alpha}(S) \\right]$$

    Parameters
    ----------
    regressor : object
        A regressor object that has a predict method.
    alpha : float
        The significance level used in the prediction interval calculation.
    Attributes
    ----------
    scores : ndarray
        The conformity scores of the calibration data.
    quantile : float
        The $(1-\\alpha)$ empirical quantile of the conformity scores.
    """
    def __init__(self, regressor, alpha=0.05):
        self.regressor = regressor
        self.alpha = alpha

    def fit(self, X, y):
        """
        Calibrates the conformal predictor using the provided calibration set.
        Specifically, the fit method learns $$q_{1-\\alpha}(S)$$ where $q_{1-\\alpha}(S)$
        is the $(1-\\alpha)$ empirical quantile of the conformity scores
        $$  S = \\left\\{ |y_i - \\hat{u}(x_i)| \\right\\} \\cup \\{ \\infty \\} $$

        Parameters
        ----------
        X : ndarray
            The input data for calibration.
        y : ndarray
            The output data for calibration.
        """
        self.regressor.fit(X, y)
        self.scores = self.regressor.predict(X)
        self.quantile = np.quantile(np.abs(self.scores - y), 1 - self.alpha)

    def predict(self, X):
        """
        Predicts the output for the given input X and provides a prediction interval.
        $$\\hat{C_\\alpha}(x) = \\left[ \\hat{u}(x) \\pm q_{1-\\alpha}(S) \\right]$$
        Parameters
        ----------
        X : ndarray
            The input data for which to predict the output.
        Returns
        -------
        tuple
            A tuple containing the prediction (1D ndarray) and the lower (1D ndarray) and upper bounds (1D ndarray) of the prediction interval.
        """
        y_pred = self.regressor.predict(X)
        y_lower = y_pred - self.quantile
        y_upper = y_pred + self.quantile
        return y_pred, y_lower, y_upper