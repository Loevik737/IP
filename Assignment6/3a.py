
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import math as m
from PIL import Image
from scipy.ndimage.morphology import binary_erosion, binary_dilation

#function that makes a binary image based on treshold, its set to 20 now
def binImg(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] > 20:
                img[i][j] = 1
            else:
                img[i][j] = 0
    return img

#importing the image and making sure its a binary image
img = Image.open('images/noisy.tiff')
img = np.array(img,dtype=np.uint8) 
img = binImg(img)


struct = np.ones((15,15))
struct2 = np.ones((26,26))

#First dilating the image to remove holes, then removing the noice. 
img2 = binary_dilation(img,structure=struct).astype(img.dtype)
img2 = binary_erosion(img2,structure=struct2).astype(img.dtype)

plt.figure()
plt.imshow(img,cmap=plt.cm.gray)
plt.show()

plt.imshow(img2,cmap=plt.cm.gray)
plt.show()
