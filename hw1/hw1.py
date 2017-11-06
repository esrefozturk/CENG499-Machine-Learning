import numpy as np

FILE = 'pima-indians-diabetes.csv'


def get_data():
    lines = [i.strip().split(',') for i in open(FILE).read().strip().split('\n')]
    X = []
    Y = []
    for line in lines:
        X.append([1] + lines[:-1])
        Y.append(line[-1])
    trainX = np.array(X[:668])
    testX = np.array(X[668:])
    trainY = np.array(Y[:668])
    testY = np.array(Y[668:])
    return trainX, testX, trainY, testY


def main():
    print get_data()


if __name__ == '__main__':
    main()
