import numpy as np

_h = 1e-4 #좋은 결과를 얻는다고 알려진 미세한 값

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

def softmax(array):
    constant = np.max(array)
    expArray = np.exp(array - constant) #오버플로우 방지
    expSum = np.sum(expArray)
    result = expArray / expSum

    return result

def mean_squared_error(y, true_table):
    return 0.5 * np.sum((y - true_table)**2)

def cross_entropy_error(y, true_table):
    delta = 1e-7 #np.log에 0이 들어가면 -inf되어서 계산을 진행할 수 없게 됨 그래서 0이 절대 안되게 이 미세한 값을 더해줌
    return -np.sum(true_table * np.log(y + delta))

def numerical_differentiation(function, x):
    f = function
    result = (f(x + _h) - f(x - _h)) / (2 * _h)
    return result

def numerical_gradient(function, x):
    gradient = np.zeros_like(x)

    for index in range(x.size):
        #f(x + h)
        _x = x[index] + _h
        fxh1 = function(_x)

        #f(x - h)
        _x = x[index] - _h
        fxh2 = function(_x)

        gradient[index] = (fxh1 - fxh2) / (2 * _h)

    return gradient

def gradient_descent(function, x, learning_rate = 0.01, step_number = 100):
    _x = x

    for step in range(step_number):
        gradient = numerical_gradient(function, x)
        _x -= learning_rate * gradient

    return _x

