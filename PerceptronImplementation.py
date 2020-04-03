import numpy as np


class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        '''
        :param eta: (float) learning rate, between 0 and 1
        :param n_iter: (int) ~ epoch
        :param random_state: (int) Random number generator seed
        Atttribute:
        w_: (1d-array) Weight after fitting
        errors_: (list) Number of missclassifications each epoch
        '''
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        '''
        Fit the training data
        :param X: {array-like}, shape = [n_examples, n_features]: Vecto huan luyen, voi n_examples là số mẫu, còn cái
        n_feature đó là số lượng thuôcj tính của nó.
        :param y: array- like, shape = [n_example] là biến target value
        :return: self: object
        '''
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1]) # Cộng thêm cái w0
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        '''Calculate the net_input
        It simple is: wT*x
        X here is x
        '''
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        '''Return class label after unit step'''
        return np.where(self.net_input(X) >= 0.0, 1, -1)

