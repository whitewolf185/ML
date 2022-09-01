import numpy as np


class Layer:
    def forward(self, *args):
        pass

    def backward(self, *args):
        pass


class Softmax(Layer):
    def forward(self, z):
        self.z = z
        zmax = z.max(axis=1, keepdims=True)
        expz = np.exp(z - zmax)
        Z = expz.sum(axis=1, keepdims=True)
        return expz / Z

    def backward(self, dp):
        p = self.forward(self.z)
        pdp = p * dp
        return pdp - p * pdp.sum(axis=1, keepdims=True)


class CrossEntropyLoss(Layer):
    def forward(self,p,y):
        self.p = p
        self.y = y
        p_of_y = p[np.arange(len(y)), y]
        log_prob = np.log(p_of_y)
        return -log_prob.mean()

    def backward(self,loss):
        dlog_softmax = np.zeros_like(self.p)
        dlog_softmax[np.arange(len(self.y)), self.y] -= 1.0/len(self.y)
        return dlog_softmax / self.p


class BinaryCrossEntropy(Layer):
    def forward(self, p, y):
        y = y.reshape((y.shape[0], 1))
        self.p = p
        self.y = y
        res = y * np.log(p) + (1 - y) * np.log(1 - p)
        return -np.mean(res)

    def backward(self, loss):
        res = (self.p - self.y) / (self.p * (1 - self.p))
        return res / self.p.shape[0]


class Linear(Layer):
    def __init__(self, nin, nout):
        sigma = 1.0 / np.sqrt(2.0 * nin)
        self.W = np.random.normal(0, sigma, (nout, nin))
        self.b = np.zeros((1, nout))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W.T) + self.b

    def backward(self, dz):
        dx = np.dot(dz, self.W)
        dW = np.dot(dz.T, self.x)
        db = dz.sum(axis=0)
        self.dW = dW
        self.db = db
        return dx

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db


class Tanh(Layer):
    def forward(self,x):
        y = np.tanh(x)
        self.y = y
        return y
    def backward(self,dy):
        return (1.0-self.y**2)*dy


class ReLU(Layer):
    def forward(self, x):
        self.x = x
        return x * (x > 0)

    def backward(self, dy):
        return (1. * (self.x > 0)) * dy


class Sigmoid(Layer):
    def forward(self, x):
        self.y = 1.0 / (1.0 + np.exp(-x))
        return self.y

    def backward(self, dy):
        return self.y * (1.0 - self.y) * dy


import matplotlib.pyplot as plt
class Net:
    def __init__(self, loss_function=CrossEntropyLoss()):
        self.layers = []
        self.loss_func = loss_function

# ----Net's standart methods----
    def add(self,l):
        self.layers.append(l)

    def forward(self,x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def backward(self,z):
        for l in self.layers[::-1]:
            z = l.backward(z)
        return z

    def update(self,lr):
        for l in self.layers:
            if 'update' in l.__dir__():
                l.update(lr)
# ----end Net's standart methods----

# ----loss functions----
    def forward_loss(self, x, y):
        p = self.forward(x)
        return self.loss_func.forward(p, y)

    def backward_loss(self, l):
        dp = self.loss_func.backward(l)
        return self.backward(dp)

    def _update_dry(self, x, y, step):
        self.update(step)
        loss = self.forward_loss(x, y)
        self.update(-step)
        return loss
# ----end loss functions----

# ----Net train----
    def _is_less_update_dry(self, x_dry, y_dry, step, l_dry, r_dry):
        lhs = (2.0 * l_dry + r_dry) / 3.0
        rhs = (l_dry + 2.0 * r_dry) / 3.0

        loss_lhs = self._update_dry(x_dry, y_dry, step * lhs)
        loss_rhs = self._update_dry(x_dry, y_dry, step * rhs)

        return loss_lhs < loss_rhs

    def train_epoch(self, epoch_train_x, train_labels, batch_size=100, step = 1e-7):
        for index in range(0, len(epoch_train_x), batch_size):
            xb = epoch_train_x[index:index + batch_size]
            yb = train_labels[index:index + batch_size]

            loss = self.forward_loss(xb, yb)
            self.backward_loss(loss)

            l = 1.0
            r = 5e2
            while r - l < 0.01:
                lhs = (2.0 * l + r) / 3.0
                rhs = (l + 2.0 * r) / 3.0
                if self._is_less_update_dry(xb,yb,step, lhs, rhs):
                    r = rhs
                else:
                    l = lhs
            self.update(r * step)


class SoftMarginSVM(Net):
    def __init__(self, nin, alpha):
        super().__init__()
        self.alpha = alpha
        sigma = 1.0 / np.sqrt(nin)
        self.W = np.random.normal(0., sigma, (1, nin + 1))

    def forward(self, x):
        z = np.dot(x, self.W.T)
        return z

    # Добавляет столбец из единиц
    def add_ones(self, x):
        ones = np.ones((x.shape[0], 1))
        return np.hstack((x, ones))

    def predict(self, x):
        res = self.forward(self.add_ones(x))
        return np.where(res < 0, 0, 1)

    def train_epoch(self, x, y, batch_size=100, step=1e-7):
        x = self.add_ones(x)
        y = np.where(y > 0, 1, -1)
        for i in range(0, len(x), batch_size):
            xb = x[i:i + batch_size]
            yb = y[i:i + batch_size]

            pred = self.forward(xb)
            grad = self.alpha * self.W
            for i in range(len(xb)):
                if (yb[i] * pred[i] < 1):
                    grad -= yb[i] * xb[i]
            self.W -= step * grad