%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import math as m
from PIL import Image
from scipy import signal

#importing image as array
img = Image.open('images/elaine.tiff')
img = np.array(img,dtype=np.float32)

#function that is later used to pad the image
def pad(vec,width,iaxis,kwargs):
    vec[:width[0]] = 0
    vec[-width[1]:] = 0
    return  vec

#declaring and rotating the kernel for horizontal gradient
kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
kernel = np.rot90(kernel,2)

#declaring adn rotating the kernel for vertical gradient
kernel2 = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
kernel2 = np.rot90(kernel2,2)

#padding the image
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

#the magnitude of the image isreturned by using the horizontal and vertical gradients in the
# formula m.sqrt((hor[i][j]**2)+(vert[i][j]**2))
def getMagnitude(hor,vert):
    retImg = np.zeros((hor.shape[0],hor.shape[1]),'float32')
    for i in range(pad_width,hor.shape[0]-pad_width):
        for j in range(pad_width,hor.shape[1]-pad_width):
            retImg[i][j] = m.sqrt((hor[i][j]**2)+(vert[i][j]**2))
    return retImg

hor = convolute(img,kernel,pad_width,256.0)
vert = convolute(img,kernel2,pad_width,256.0)

plt.figure()
plt.imshow(getMagnitude(hor,vert), cmap=plt.cm.gray)
plt.show()
plt.imshow(hor, cmap=plt.cm.gray)
plt.show()
plt.imshow(vert, cmap=plt.cm.gray)
plt.show()
