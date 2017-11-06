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
    return (1 / (1 - np.exp(-t))).reshape(-1, 1)


def loss(x, w, y):
    s = sigmoid(x, w)

    return -(np.dot(y.T, np.log(s)) + np.dot((1 - y).T, (1 - s))) / float(x.shape[0])


def minimize_loss(x, w, y, num_iters=10000, eta=3e-4):
    for i in range(num_iters):
        s = sigmoid(x, w)

        dw = np.dot(x.T, (s - y)) / float(x.shape[0])

        w -= eta * dw
    return w


def main():
    trainX, testX, trainY, testY = get_data()
    trainX = scale(trainX)
    trainX = np.append(np.ones((trainX.shape[0], 1)), trainX, axis=1)

    testX = scale(testX)
    testX = np.append(np.ones((testX.shape[0], 1)), testX, axis=1)

    np.random.seed(499)
    w = np.random.random_sample(trainX.shape[1]).reshape(-1, 1)
    w = minimize_loss(trainX, w, trainY)

    asdas = sigmoid(testX, w) >= .5

    print np.mean(asdas == testY)


if __name__ == '__main__':
    main()
