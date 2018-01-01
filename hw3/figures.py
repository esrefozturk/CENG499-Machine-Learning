import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

X = np.load('FaceImages.npy')

L = []
R = []

for i in xrange(4):
	L.append(X[i,:].reshape(231,195))
R.append(np.hstack(L))
L = []

for i in xrange(4,8):
	L.append(X[i,:].reshape(231,195))
R.append(np.hstack(L))
L = []

for i in xrange(8,12):
	L.append(X[i,:].reshape(231,195))
R.append(np.hstack(L))
L = []

for i in xrange(12,16):
	L.append(X[i,:].reshape(231,195))
R.append(np.hstack(L))

A = np.vstack(R)

imgplot = plt.imshow(A, cmap='gray')
plt.show()

###############################################

R = []
L = []
for i in xrange(8):
	for j in xrange(8):
		img = np.ones((8,8))
		L.append(img)
	R.append(np.hstack(L))
	L = []
A = np.vstack(R)

imgplot = plt.imshow(A, cmap='gray')
plt.show()






