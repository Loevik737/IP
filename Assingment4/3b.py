%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#importing image as array
img = Image.open('images/fishingboat.tiff')
img = np.array(img)

#a function that normalize the image values to range between [0,1]
#change the pk value for if your image is not 8 bit. Also notise that the normalized array has float as datatype
def normalize(img):
    normalized = np.full((img.shape[0],img.shape[1]),0,dtype=np.float32)
    pk = np.iinfo(np.uint8).max
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            normalized[i][j] = float(img[i][j]) / float(pk)
    return normalized

#function that return the image with a gamma value applied to each pixle
def gammaTrans(img,gamma):
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            img[i][j] = img[i][j]**gamma
    return img


normalized = normalize(img)
gammaTransformed = gammaTrans(normalized,0.1)

plt.figure()
plt.imshow(img, cmap=plt.cm.gray)
plt.show()
plt.imshow(gammaTransformed,cmap=plt.cm.gray)
plt.show()
