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
                new_centers.append(SUMS[i]/LENS[i])
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


def main():
    images = np.load('FaceImages.npy')

    #images = images[:10]

    centers = findClusterCenters(images, 2)


    images = kmeansCompress(images, centers)

    for i in range(len(images)):
        show(images[i])


if __name__ == '__main__':
    main()
