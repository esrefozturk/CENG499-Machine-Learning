import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show(img):
	imgplot = plt.imshow(img.reshape(231,195), cmap='gray')
	plt.show()
