from sklearn.linear_model import LogisticRegression
import numpy as np

def get_lr_probs_params(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fit a logistic regression model on provided input X and labels Y.
    Return class probabilities for all X and params calculated for model.
    """
    lr = LogisticRegression()
    lr.fit(X, y)
    params={'lr_coefficients': lr.coef_.tolist(), 'lr_intercept': lr.intercept_.item()}
    probs = lr.predict_proba(X)
    hit_probs = probs[:,-1]
    return hit_probs, params
