import matplotlib.pyplot as plt
import numpy as np

FILE = 'pima-indians-diabetes.csv'


def get_data():
    data = np.loadtxt(FILE, delimiter=",")

    train = data[:568]
    validation = data[568:668]
    test = data[668:]

    trainX = scale(train[:, :-1])
    trainY = train[:, -1].reshape(-1, 1)

    validationX = scale(validation[:, :-1])
    validationY = validation[:, -1].reshape(-1, 1)

    testX = scale(test[:, :-1])
    testY = test[:, -1].reshape(-1, 1)

    return trainX, trainY, validationX, validationY, testX, testY


def scale(x):
    mi = np.amin(x, axis=0).reshape(-1, 1).T
    ma = np.amax(x, axis=0).reshape(-1, 1).T
    scaled = (x - mi) / (ma - mi)
    return np.append(np.ones((scaled.shape[0], 1)), scaled, axis=1)


def sigmoid(x, w):
    t = np.dot(x, w)
    return (1 / (1 + np.exp(-t))).reshape(-1, 1)


def minimize_loss(trainX, trainY, validationX, validationY, testX, testY, w, num_iters=10000, eta=1e-1):
    train_losses = []
    train_accuracies = []

    validation_losses = []
    validation_accuracies = []

    test_losses = []
    test_accuracies = []

    for i in range(num_iters):
        s = sigmoid(trainX, w)
        sV = sigmoid(validationX, w)
        sT = sigmoid(testX, w)

        dw = np.dot(trainX.T, (s - trainY)) / float(trainX.shape[0])

        train_losses.append(
            (-(np.dot(trainY.T, np.log(s)) + np.dot((1 - trainY).T, (1 - s))) / float(trainX.shape[0]))[0])
        train_accuracies.append(np.mean((s >= .5) == trainY))

        validation_losses.append((-(np.dot(validationY.T, np.log(sV)) + np.dot((1 - validationY).T, (1 - sV))) / float(
            validationX.shape[0]))[0])
        validation_accuracies.append(np.mean((sV >= .5) == validationY))

        test_losses.append((-(np.dot(testY.T, np.log(sT)) + np.dot((1 - testY).T, (1 - sT))) / float(
            testX.shape[0]))[0])
        test_accuracies.append(np.mean((sT >= .5) == testY))

        w -= eta * dw

    plt.plot(range(num_iters), train_losses, label='{eta} - train loss'.format(eta=eta))
    plt.plot(range(num_iters), train_accuracies, label='{eta} - train accuracy'.format(eta=eta))

    # plt.plot(range(num_iters), validation_losses, label='{eta} - validation loss'.format(eta=eta))
    # plt.plot(range(num_iters), validation_accuracies, label='{eta} - validation accuracy'.format(eta=eta))

    plt.plot(range(num_iters), test_losses, label='{eta} - test loss'.format(eta=eta))
    plt.plot(range(num_iters), test_accuracies, label='{eta} - test accuracy'.format(eta=eta))

    return w


def main():
    trainX, trainY, validationX, validationY, testX, testY = get_data()

    for eta in [1e-1]:
        np.random.seed(499)
        w = np.random.random_sample(trainX.shape[1]).reshape(-1, 1)

        w = minimize_loss(trainX, trainY, validationX, validationY, testX, testY, w, eta=eta)

        print np.mean((sigmoid(testX, w) >= .5) == testY)

        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
