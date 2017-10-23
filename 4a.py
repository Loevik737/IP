%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import math as m
from PIL import Image
from scipy import signal

img = Image.open('images/fishingboat.tiff')
img = np.array(img,dtype=np.float32)
img = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],dtype=np.float32)
def pad(vec,width,iaxis,kwargs):
    vec[:width[0]] = 0
    vec[-width[1]:] = 0
    return  vec

kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])
kernel = np.rot90(kernel,2)
pad_width = m.floor(float(kernel.shape[0])/2)
img = np.lib.pad(img,pad_width,pad)

def convolute(img,kernel,pad_width):
    for i in range(pad_width,img.shape[0]-pad_width):
        for j in range(pad_width,img.shape[1]-pad_width):
            val = 0.0
            for k in range(0,kernel.shape[0]):
                for l in range(0,kernel.shape[1]):
                    val +=  int(kernel[k][l] * img[i-pad_width+k][j-pad_width+l])
            img[i][j] = val/9.0
    return img
print(convolute(img,kernel,pad_width))
plt.figure()
plt.imshow(convolute(img,kernel,pad_width), cmap=plt.cm.gray)
plt.show()
