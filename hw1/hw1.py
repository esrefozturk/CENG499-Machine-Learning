import numpy as np

FILE = 'pima-indians-diabetes.csv'


def get_data():
    lines = [i.strip().split(',') for i in open(FILE).read().strip().split('\n')]
    X = []
    Y = []
    for line in lines:
        X.append([1] + line[:-1])
        Y.append(line[-1])
    trainX = np.array(X[:668])
    testX = np.array(X[668:])
    trainY = np.array(Y[:668])
    testY = np.array(Y[668:])
    return trainX, testX, trainY, testY


def sigmoid(x, w):
    t = np.dot(w.T, x)
    return 1 / (1 - np.exp(-t))


def main():
    trainX, testX, trainY, testY = get_data()
    np.random.seed(499)
    w = np.concatenate([[0], np.random.random_sample(trainX.shape[1])])


if __name__ == '__main__':
    main()
