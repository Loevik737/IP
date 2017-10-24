%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import math as m
from PIL import Image
from scipy import signal

img = Image.open('images/lake.tiff')
img = np.array(img,dtype=np.float32)

plt.figure()
plt.imshow(img)
plt.show()

kernel = np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]])
kernel = np.rot90(kernel,2)
pad_width = m.floor(float(kernel.shape[0])/2)

def convolute(img,kernel,pad_width,scalar,color):
    if color:
        retImg = retImg = np.zeros((img.shape[0],img.shape[1],3),'float32')
        for c in range(0,3):
            for i in range(pad_width,img.shape[0]-pad_width):
                for j in range(pad_width,img.shape[1]-pad_width):
                    val = 0.0
                    for k in range(0,kernel.shape[0]):
                        for l in range(0,kernel.shape[1]):
                            val +=  int(kernel[k][l] * img[i-pad_width+k][j-pad_width+l][c])
                    retImg[i][j][c] = val/scalar
        return retImg
    else:
        retImg = np.zeros((img.shape[0],img.shape[1]),'float32')
        for i in range(pad_width,img.shape[0]-pad_width):
            for j in range(pad_width,img.shape[1]-pad_width):
                val = 0.0
                for k in range(0,kernel.shape[0]):
                    for l in range(0,kernel.shape[1]):
                        val +=  int(kernel[k][l] * img[i-pad_width+k][j-pad_width+l])
                retImg[i][j] = val/scalar
        return retImg

plt.figure()
plt.imshow(convolute(img,kernel,pad_width,256.0,True))
plt.show()
