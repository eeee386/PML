import numpy as np


class AdalineGD(object):
    """
    ADAptive LInear NEuron classifier, Batch gradient descent

    Parameters
    -----------
    :param eta: float, Learning rate (between 0.0 and 1.0)
    :param n_iter: int, Passes over the training data set
    :param random_state: int, Random number seed for random weight

    Attributes
    -----------
    :param w_: 1d-array, Weights after fitting
    :param cost: list, Sum-of-squares cost function value in each epoch
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit training data

        :param X: {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and n_features is the number of features
        :param y: array-like, shape: [n_samples]
        Target values
        :return: self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y-output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate the net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X))>=0.0, 1, -1)


class AdalineSGD(object):
    """
    ADAptive LInear NEuron classifier, Stochastic gradient descent

    Parameters
    -----------
    :param eta: float, Learning rate (between 0.0 and 1.0)
    :param n_iter: int, Passes over the training data set
    :param random_state: int, Random number seed for random weight
    :param shuffle: bool (default True), Shuffles the data every epoch if True to prevent cycles

    Attributes
    -----------
    :param w_: 1d-array, Weights after fitting
    :param cost: list, Sum-of-squares cost function value in each epoch
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=None, shuffle=True):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.w_initialized = False
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit training data

        :param X: {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and n_features is the number of features
        :param y: array-like, shape: [n_samples]
        Target values
        :return: self : object
        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X,y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """
        Fit training data without reinitializing the weights
        :param X:
        :param y:
        :return:
        """
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(X, y)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Initialize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1+m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = target-output
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost

    def net_input(self, X):
        """Calculate the net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X))>=0.0, 1, -1)
