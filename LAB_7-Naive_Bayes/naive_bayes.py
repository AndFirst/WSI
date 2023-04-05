import numpy as np
from numpy import ndarray


def gaussian_density(x: ndarray, mean: float, var: float) -> ndarray:
    const = 1 / np.sqrt(var * 2 * np.pi)
    exponent = -0.5 * (np.square(x - mean) / var)
    return const * np.exp(exponent)


class NaiveBayes:
    def __init__(self) -> None:
        self._mean = None
        self._variance = None
        self._priors = None
        self._n_classes = None

    def fit(self, x: np.array, y: np.array) -> None:
        n_samples, n_features = x.shape
        self._n_classes = len(np.unique(y))

        self._mean = np.zeros((self._n_classes, n_features))
        self._variance = np.zeros((self._n_classes, n_features))
        self._priors = np.zeros(self._n_classes)

        for i in range(self._n_classes):
            x_i = x[y == i]
            self._mean[i, :] = np.mean(x_i, axis=0)
            self._variance[i, :] = np.var(x_i, axis=0)
            self._priors[i] = x_i.shape[0] / n_samples

    def predict(self, x: ndarray) -> ndarray:
        y_pred = np.array([self.__predict_class(x_i) for x_i in x])
        return y_pred

    def __predict_class(self, x: ndarray) -> ndarray[int]:
        posteriors = list()

        for i in range(self._n_classes):
            mean = self._mean[i]
            variance = self._variance[i]
            prior = np.log(self._priors[i])

            posterior = np.sum(np.log(gaussian_density(x, mean, variance)))
            posterior += prior
            posteriors.append(posterior)

        return np.argmax(posteriors)
