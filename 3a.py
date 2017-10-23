%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

img = Image.open('images/fishingboat.tiff')
img = np.array(img)

def transform(img):
    pk = np.iinfo(np.uint8).max
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            img[i][j] = int(pk) - img[i][j]
    return img

plt.figure()
plt.imshow(img, cmap=plt.cm.gray)
plt.show()
plt.imshow(transform(img), cmap=plt.cm.gray)
plt.show()
