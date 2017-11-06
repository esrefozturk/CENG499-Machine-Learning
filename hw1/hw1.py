import numpy as np

FILE = 'pima-indians-diabetes.csv'


def get_data():
    data = np.loadtxt(FILE, delimiter=",")

    train = data[:558]
    validation = data[558:668]
    test = data[668:]

    trainX = scale(train[:, :-1])
    trainY = train[:, -1].reshape(-1, 1)

    validationX = scale(validation[:, :-1])
    validationY = validation[:, -1].reshape(-1, 1)

    testX = scale(test[:, :-1])
    testY = test[:, -1].reshape(-1, 1)

    np.random.seed(499)
    w = np.random.random_sample(trainX.shape[1]).reshape(-1, 1)

    return trainX, trainY, validationX, validationY, testX, testY, w


def scale(x):
    mi = np.amin(x, axis=0)
    ma = np.amax(x, axis=0)
    scaled = (x - mi) / (ma - mi)
    return np.append(np.ones((scaled.shape[0], 1)), scaled, axis=1)


def sigmoid(x, w):
    t = np.dot(x, w)
    return (1 / (1 - np.exp(-t))).reshape(-1, 1)


def minimize_loss(trainX, trainY, validationX, validationY, w, num_iters=10000, eta=3e-4):
    losses = []

    for i in range(num_iters):
        s = sigmoid(trainX, w)

        dw = np.dot(trainX.T, (s - trainY)) / float(trainX.shape[0])
        losses.append(-(np.dot(trainY.T, np.log(s)) + np.dot((1 - trainY).T, (1 - s))) / float(trainX.shape[0]))

        w -= eta * dw
    return w


def main():

    trainX, trainY, validationX, validationY, testX, testY, w = get_data()

    w = minimize_loss(trainX, trainY, validationX, validationY, w)

    asdas = sigmoid(testX, w) >= .5

    print np.mean(asdas == testY)


if __name__ == '__main__':
    main()
