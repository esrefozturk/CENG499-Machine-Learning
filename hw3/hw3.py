from random import randint

import numpy as np

from function import show

CENTERS = []


def random_k_points(K):
    return sorted([randint(0, 255) for i in range(K)])


def findClusterCenters(images, K):
    (a, b) = images.shape
    images = [i[0] for i in images.reshape(a * b, 1).tolist()]

    centers = random_k_points(K)

    while 1:
        print centers
        SUMS = {c: 0 for c in centers}
        LENS = {c: 0 for c in centers}

        def assign1(x):
            SUMS[centers[(np.abs(np.array(centers) - x)).argmin()]] += x
            LENS[centers[(np.abs(np.array(centers) - x)).argmin()]] += 1
            return x

        f = np.vectorize(assign1)
        f(images)
        new_centers = []
        for i in SUMS:
            if LENS[i]:
                new_centers.append(SUMS[i] / LENS[i])
            else:
                new_centers.append(i)
        new_centers = sorted(new_centers)
        if centers == new_centers:
            break
        centers = new_centers

    return centers


def kmeansCompress(images, centers):
    def compress(x):
        return centers[(np.abs(np.array(centers) - x)).argmin()]

    f = np.vectorize(compress)
    return f(images)


def kmeans():
    images = np.load('FaceImages.npy')

    # images = images[:10]

    centers = findClusterCenters(images, 2)

    images = kmeansCompress(images, centers)

    for i in range(len(images)):
        show(images[i])


def findPrincipalComponents(images, K):
    images = (images - np.mean(images, axis=0)) / np.std(images, axis=0)
    vals, vec = np.linalg.eig(np.dot(images, images.T))
    return np.dot(images.T, vec)[:, :K]


def pcaCompress(images, pca):
    c = np.dot(images, pca)
    return np.dot(c, pca.T)


def main():
    images = np.load('FaceImages.npy')
    pca = findPrincipalComponents(images, 2)

    images = pcaCompress(images, pca)

    show(images[0])

    show(images[1])

    show(images[2])


if __name__ == '__main__':
    main()
