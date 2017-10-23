%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

img = Image.open('images/lochness.tiff')
img = np.array(img)

def averageGray(img):
    gray = np.full((img.shape[0],img.shape[1]),0,dtype=np.uint8)
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            gray[i][j] = np.average(img[i][j])
    return gray

def luminGray(img):
    gray = np.full((img.shape[0],img.shape[1]),0,dtype=np.uint8)
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            value = 0
            red = 0.2126
            green = 0.7152
            blue = 0.0722
            for k in range(0,3):
                if k == 0:
                    value += red*img[i][j][k]
                elif k == 1:
                    value += green*img[i][j][k]
                else:
                    value += blue*img[i][j][k]
            gray[i][j]  = value
    return gray
plt.figure()
plt.imshow(luminGray(img), cmap=plt.cm.gray)
plt.show()
plt.imshow(averageGray(img), cmap=plt.cm.gray)
plt.show()
