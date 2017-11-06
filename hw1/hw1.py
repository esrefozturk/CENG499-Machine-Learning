import numpy as np

FILE = 'pima-indians-diabetes.csv'


def get_data():
    lines = [i.strip().split(',') for i in open(FILE).read().strip().split('\n')]
    X = []
    Y = []
    for line in lines:
        X.append(line[:-1])
        Y.append(line[-1])
    trainX = np.array(X[:668], dtype=np.float64)
    testX = np.array(X[668:], dtype=np.float64)
    trainY = np.array(Y[:668], dtype=np.float64).reshape(-1, 1)
    testY = np.array(Y[668:], dtype=np.float64).reshape(-1, 1)
    return trainX, testX, trainY, testY


def scale(x):
    mi = np.amin(x, axis=0)
    ma = np.amax(x, axis=0)
    return (x - mi) / (ma - mi)


def sigmoid(x, w):
    t = np.dot(x, w)
    return 1 / (1 - np.exp(-t))


def minimize_loss(x, w, y, num_iters=1000, eta=0.001):
    for i in range(num_iters):
        s = sigmoid(x, w).reshape(-1, 1)

        dw = np.sum((y - s) * x) / float(x.shape[0])

        w -= eta * dw
    return w


def main():
    trainX, testX, trainY, testY = get_data()
    trainX = scale(trainX)
    trainX = np.append(np.ones((trainX.shape[0], 1)), trainX, axis=1)

    np.random.seed(499)
    w = np.random.random_sample(trainX.shape[1]).reshape(-1, 1)
    w = minimize_loss(trainX, w, trainY)


if __name__ == '__main__':
    main()
