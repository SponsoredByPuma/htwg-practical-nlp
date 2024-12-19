import numpy as np


class LogisticRegression:
    """Logistic Regression class for NLP tasks.

    Attributes:
        theta (np.ndarray | None): the weights of the model, None before training
        cost (float | None): the cost of the model, None before training
        learning_rate (float): the learning rate of the model
        n_iter (int): the maximum number of iterations for the training
    """

    def __init__(self, learning_rate: float = 1e-9, n_iter: int = 1500) -> None:
        """Initialize the LogisticRegression class."""
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.theta: np.ndarray = None
        self.cost: float | None = None

    def _initialize_weights(self, n_features: int) -> None:
        """Initialize the weights of the model."""
        self.theta = np.zeros((n_features, 1))

    def _sigmoid(self, z: np.ndarray | float) -> np.ndarray | float:
        """Compute the sigmoid of z."""
        return 1 / (1 + np.exp(-z))

    def _cost_function(self, y: np.ndarray, h: np.ndarray) -> None:
        """Compute the cost in the current iteration."""
        m = y.shape[0]
        epsilon = 1e-15  # Prevent log(0)
        self.cost = -(1 / m) * np.sum(
            y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon)
        )

    def _update_weights(self, X: np.ndarray, y: np.ndarray, h: np.ndarray) -> None:
        """Update the weights of the model."""
        m = X.shape[0]
        gradient = (1 / m) * np.dot(X.T, (h - y))
        self.theta -= self.learning_rate * gradient

    def _gradient_descent(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray | None, float | None]:
        """Perform gradient descent with training data X and labels y."""
        self._initialize_weights(X.shape[1])
        for _ in range(self.n_iter):
            z = np.dot(X, self.theta)
            h = self._sigmoid(z)
            self._cost_function(y, h)
            self._update_weights(X, y, h)
        return self.theta, self.cost

    def fit(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray | None, float | None]:
        """Perform gradient descent with training data X and labels y."""
        assert X.shape[0] == y.shape[0], "Number of samples in X and y must be equal."
        assert X.ndim == 2, "X must be a 2-dimensional array."
        assert y.ndim == 2, "y must be a 2-dimensional array."
        assert y.shape[1] == 1, "y must be a column vector."

        return self._gradient_descent(X, y)

    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for the given samples."""
        assert (
            self.theta is not None
        ), "Model must be trained before making predictions."
        return self._sigmoid(np.dot(X, self.theta))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict the labels for the given samples."""
        probabilities = self.predict_prob(X)
        return np.where(probabilities >= threshold, 1, 0)
