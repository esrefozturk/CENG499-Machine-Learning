import matplotlib.pyplot as plt
import numpy as np


def random_k_points(K):
    return np.sort(np.random.randint(0, 255, K))


def findClusterCenters(images, K):
    images = images.reshape(-1)
    centers = random_k_points(K)

    while 1:
        new_centers = np.copy(centers)
        d = np.abs(centers.reshape(-1, 1) - images)
        d = np.argmin(d, axis=0)
        for i in range(K):
            if not np.any(d == i):
                continue
            new_centers[i] = np.mean(images[d == i]).astype(int)
        new_centers = np.sort(new_centers)
        if np.array_equal(centers, new_centers):
            break
        centers = new_centers

    return centers


def kmeansCompress(images, centers):
    images = images.reshape(-1)
    d = np.abs(centers.reshape(-1, 1) - images)
    d = np.argmin(d, axis=0)

    s = np.zeros(images.shape[0])

    for i, c in enumerate(centers):
        s += (d == i) * c

    return s.reshape(-1, 45045)


def asdas(centers, K1, K2):
    R = []
    L = []
    for i in xrange(K1):
        for j in xrange(K2):
            img = np.ones((8, 8)) * centers[K2 * i + j]
            L.append(img)
        R.append(np.hstack(L))
        L = []
    A = np.vstack(R)

    plt.imshow(A, cmap='gray')
    plt.show()


def kmeans(images, K, K1, K2):
    centers = findClusterCenters(images, K)
    # asdas(centers, K1, K2)

    images = kmeansCompress(images, centers)

    # show(images[0])
    # show(images[1])

    return images


def findPrincipalComponents(images, K):
    images = (images - np.mean(images, axis=0)) / np.std(images, axis=0)
    vals, vec = np.linalg.eig(np.dot(images, images.T))
    return np.dot(images.T, vec)[:, :K]


def pcaCompress(images, pca):
    images = (images - np.mean(images, axis=0)) / np.std(images, axis=0)
    c = np.dot(images, pca)
    return np.dot(c, pca.T)


def qwe(X):
    L = []
    R = []

    for i in xrange(4):
        L.append(X[i, :].reshape(231, 195))
    R.append(np.hstack(L))
    L = []

    for i in xrange(4, 8):
        L.append(X[i, :].reshape(231, 195))
    R.append(np.hstack(L))
    L = []

    for i in xrange(8, 12):
        L.append(X[i, :].reshape(231, 195))
    R.append(np.hstack(L))
    L = []

    for i in xrange(12, 16):
        L.append(X[i, :].reshape(231, 195))
    R.append(np.hstack(L))

    A = np.vstack(R)

    plt.imshow(A, cmap='gray')
    plt.show()


def pca(images, K):
    p = findPrincipalComponents(images, K)

    X = np.array([])

    for i in range(K):
        X = np.append(X, p.T[i])
    # qwe(X.reshape(-1,45045))

    images = pcaCompress(images, p)

    # show(images[0])
    # show(images[1])

    return images


def main():
    images = np.load('FaceImages.npy')

    kmeans(np.copy(images), 16, 4, 4)
    kmeans(np.copy(images), 32, 4, 8)
    kmeans(np.copy(images), 64, 8, 8)
    pca(np.copy(images), 16)
    pca(np.copy(images), 32)
    pca(np.copy(images), 64)


if __name__ == '__main__':
    main()
