import numpy as np


def random_k_points(K):
    return np.sort(np.random.randint(0, 255, K))


def fast_assign_points(centers, d):
    assigned_centers = {i: [] for i in centers}
    for i in d:
        n = assigned_centers.keys()[0]
        for j in assigned_centers:
            if abs(j - i) < abs(n - i):
                n = j
        assigned_centers[n].append(i)
    return assigned_centers


def fast_get_new_centers(assigned_centers, d):
    centers = []
    for c in assigned_centers:
        s = 0
        l = 0
        for i in assigned_centers[c]:
            s += d[i] * i
            l += d[i]
        if l:
            centers.append(s / l)
        else:
            centers.append(c)
    return centers


def fast_findClusterCenters(images, K):
    images = images.reshape(-1).tolist()
    centers = random_k_points(K).tolist()

    d = {}
    for i in images:
        d[i] = d.get(i, 0) + 1

    while 1:
        assigned_centers = fast_assign_points(centers, d)
        new_centers = sorted(fast_get_new_centers(assigned_centers, d))
        if new_centers == centers:
            break
        centers = new_centers

    return np.array(centers)


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


def kmeans(images, K):
    centers = findClusterCenters(images, K)

    images = kmeansCompress(images, centers)

    return images


def findPrincipalComponents(images, K):
    images = (images - np.mean(images, axis=0)) / np.std(images, axis=0)
    vals, vec = np.linalg.eig(np.dot(images, images.T))
    return np.dot(images.T, vec)[:, :K]


def pcaCompress(images, pca):
    images = (images - np.mean(images, axis=0)) / np.std(images, axis=0)
    c = np.dot(images, pca)
    return np.dot(c, pca.T)


def pca(images, K):
    p = findPrincipalComponents(images, K)

    images = pcaCompress(images, p)

    return images


def main():
    images = np.load('FaceImages.npy')
    K = 8
    kmeans(np.copy(images), K)

    pca(np.copy(images), K)


if __name__ == '__main__':
    main()
