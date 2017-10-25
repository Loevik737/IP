%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#imports the image as an array
img = Image.open('images/lochness.tiff')
img = np.array(img)

#function that finds the average grayscale value of the rbg image and returns a grayscale image
def averageGray(img):
    gray = np.full((img.shape[0],img.shape[1]),0,dtype=np.uint8)
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            gray[i][j] = np.average(img[i][j])
    return gray

#Function that does the same as in averageGray but with weighted color values
def luminGray(img):
    gray = np.full((img.shape[0],img.shape[1]),0,dtype=np.uint8)
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            value = 0
            for k in range(0,3):
                if k == 0:
                    value += 0.2126*img[i][j][k]
                elif k == 1:
                    value += 0.7152*img[i][j][k]
                else:
                    value += 0.0722*img[i][j][k]
            gray[i][j]  = value
    return gray

plt.figure()
plt.imshow(averageGray(img), cmap=plt.cm.gray)
plt.show()
plt.imshow(luminGray(img), cmap=plt.cm.gray)
plt.show()
