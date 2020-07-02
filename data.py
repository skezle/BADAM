import math
import numpy as np

def reg(n, train, seed=123):

    """
    Generator for regression data from experiment in BBB
    :param iters: int, max number of iterations
    :param train: bool, whether to perform training of not
    :return: single point x, y (2-d regression problem)
    """
    np.random.seed(seed if train else None)
    x = np.arange(0.0 if train else -0.5, 0.5 if train else 1.2, 0.5/float(n) if train else 1.7/float(n))
    if train:
        np.random.shuffle(x)
    epsilon = np.random.normal(loc=0.0, scale=0.02, size=x.shape)
    y = x + 0.3 * np.sin(2 * math.pi * (x + epsilon)) + \
        0.3 * np.sin(4 * math.pi * (x + epsilon)) + \
        epsilon
    return x, y

def reg_iter(n, train, batch_size, seed=123):
    """
    Generator for regression data from experiment in BBB
    :param iters: int, max number of iterations
    :param train: bool, whether to perform training of not
    :return: single point x, y (2-d regression problem)
    """
    np.random.seed(seed if train else None)
    x = np.arange(0.0 if train else -0.5, 0.5 if train else 1.2, 0.5 / float(n) if train else 1.7 / float(n))
    np.random.shuffle(x)
    epsilon = np.random.normal(loc=0.0, scale=0.02, size=x.shape)
    y = x + 0.3 * np.sin(2 * math.pi * (x + epsilon)) + \
        0.3 * np.sin(4 * math.pi * (x + epsilon)) + \
        epsilon
    num_batches = int(math.ceil(x.shape[0] / float(batch_size)))

    #print("total number of batches: {}".format(num_batches))
    for i in range(num_batches):
        yield x[int(i * batch_size):int((i + 1) * batch_size)].reshape(-1, 1), \
              y[int(i * batch_size):int((i + 1) * batch_size)].reshape(-1, 1)