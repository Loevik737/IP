%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import math as m
from PIL import Image
from scipy import fftpack


def circle(kernel,radius,value):
    radius = radius
    cx, cy = int(m.floor(kernel.shape[0]/2)), int(m.floor(kernel.shape[1]/2)) # The center of circle
    y, x = np.ogrid[-radius: radius, -radius: radius]
    index = x**2 + y**2 <= radius**2
    kernel[cy-radius:cy+radius, cx-radius:cx+radius][index] = value
    return kernel
                                        

img = Image.open('images/moon.tiff').convert('L')
img = np.array(img,dtype=np.float32)

freq = np.fft.fft2(img,s=(2*img.shape[0],2*img.shape[1]))
freqS = np.fft.fftshift(freq)

whiteKernel = np.full((2*img.shape[0],2*img.shape[1]),1,dtype=np.uint8)
blackKernel = np.full((2*img.shape[0],2*img.shape[1]),0,dtype=np.uint8)

filteredLp = freqS * circle(blackKernel,70,1)
filteredLp = np.fft.ifftshift(filteredLp)
filteredLpS = np.fft.fftshift(filteredLp)

filteredHp = freqS * circle(whiteKernel,30,0)
filteredHp = np.fft.ifftshift(filteredHp)
filteredHpS = np.fft.fftshift(filteredHp)

blured = np.fft.ifft2(filteredLp).real
blured = blured[:img.shape[0],:img.shape[1]]

sharp = np.fft.ifft2(filteredHp).real
sharp = sharp[:img.shape[0],:img.shape[1]]
        
plt.figure()
plt.imshow(img, cmap=plt.cm.gray)
plt.show()

plt.imshow(np.log2(1+abs(freqS)),cmap=plt.cm.gray)
plt.show()

plt.imshow(np.log2(1+abs(filteredLpS)),cmap=plt.cm.gray)
plt.show()

plt.imshow(np.log2(1+abs(filteredHpS)),cmap=plt.cm.gray)
plt.show()

plt.imshow(blured,cmap=plt.cm.gray)
plt.show()

plt.imshow(sharp,cmap=plt.cm.gray)
plt.show()

