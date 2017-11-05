import numpy as np

FILE = 'pima-indians-diabetes.csv'


def get_data():
    lines = [i.strip().split(',') for i in open(FILE).read().strip().split('\n')]
    train_data = []
    test_data = []
    for line in lines:
        train_data.append(lines[:-1])
        test_data.append(line[-1])
    trainX = np.array(train_data[:668])
    testX = np.array(test_data[:668])
    trainY = np.array(train_data[668:])
    testY = np.array(test_data[668:])
    return trainX, testX, trainY, testY


def main():
    print get_data()


if __name__ == '__main__':
    main()
