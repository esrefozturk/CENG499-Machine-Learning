import numpy as np

from function import show


def random_k_points(K):
    return np.sort(np.random.randint(0, 255, K))


def findClusterCenters(images, K):
    images = images.reshape(-1)
    centers = random_k_points(K)

    while 1:
        print centers
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
        print s
        s += (d == i) * c

    return s.reshape(-1, 45045)


def kmeans(images, K):
    centers = findClusterCenters(images, K)

    images = kmeansCompress(images, centers)

    for i in range(len(images[:1])):
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

    kmeans(images, 2)

    # pca = findPrincipalComponents(images, 2)

    # images = pcaCompress(images, pca)

    # show(images[0])


if __name__ == '__main__':
    main()
