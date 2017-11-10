%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import math as m
from PIL import Image
from scipy import fftpack

def removeVertical(kernel,width,gap):
    for i in range(int(m.floor(kernel.shape[0]/2))-width,int(m.floor(kernel.shape[0]/2))+width):
        for j in range(0,len(kernel[i])):
            if j < (len(kernel[i])/2)-gap or j > (len(kernel[i])/2)+gap:
                kernel[i][j] = 0
    return kernel
def removeCircle(kernel,radius,value):

    radius = radius 
    cx, cy = int(m.floor(kernel.shape[0]/2)), int(m.floor(kernel.shape[1]/2)) # The center of circle
    y, x = np.ogrid[-radius: radius, -radius: radius]
    index = x**2 + y**2 <= radius**2
    kernel[cy-radius:cy+radius, cx-radius:cx+radius][index] = value
    if value == 0:
        removeCircle(kernel,radius-15,1)
    return kernel

img = Image.open('images/noise-a.tiff')
img = np.array(img,dtype=np.uint8)
freq = np.fft.fft2(img,s=(2*img.shape[0],2*img.shape[1]))
freqS = np.fft.fftshift(freq)

kernel = np.full((2*img.shape[0],2*img.shape[1]),1,dtype=np.uint8)
#kernel = removeCircle(kernel,110,0)
kernel = removeVertical(kernel,20,65)

filtered = freqS * kernel
filtered = np.fft.ifftshift(filtered)
filteredS = np.fft.fftshift(filtered)

img2 = np.fft.ifft2(filtered).real
img2 = img2[:img.shape[0],:img.shape[1]]

plt.figure()
plt.imshow(img, cmap=plt.cm.gray)
plt.show()

plt.imshow(np.log2(1+abs(freqS)),cmap=plt.cm.gray)
plt.show()

plt.imshow(np.log2(1+abs(filteredS)),cmap=plt.cm.gray)
plt.show()
plt.imshow(kernel,cmap=plt.cm.gray)
plt.show()

plt.imshow(img2,cmap=plt.cm.gray)
plt.show()

