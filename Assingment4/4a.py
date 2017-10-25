%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import math as m
from PIL import Image
from scipy import signal

#importing image as array
img = Image.open('images/fishingboat.tiff')
img = np.array(img,dtype=np.float32)

#function that is later used to pad the image
def pad(vec,width,iaxis,kwargs):
    vec[:width[0]] = 0
    vec[-width[1]:] = 0
    return  vec

#the kernel is declared and flipped
kernel = np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]])
#kernel = np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])
kernel = np.rot90(kernel,2)
#pad the image with the floor of the half of the kernel width
pad_width = m.floor(float(kernel.shape[0])/2)
img = np.lib.pad(img,pad_width,pad)

#function that returns the image after its convoluted. The scalar can be used for fore example averaging
def convolute(img,kernel,pad_width,scalar):
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
plt.imshow(convolute(img,kernel,pad_width,9.0), cmap=plt.cm.gray)
plt.show()
