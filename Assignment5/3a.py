%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import math as m
from PIL import Image
from scipy import fftpack

def gaussian(size, std):
    s = (size - 1) // 2
    h = np.linspace(-s, s, size)
    h = np.exp(-h**2 / (2 * std**2))
    h = h * h[np.newaxis, :].T
    sumh = h.sum()
    if 0.0 != sumh:
        h /= sumh
    return h

def dirac(size):
    kernel = np.zeros((size,size))
    kernel[size//2][size//2] = 1
    return kernel

img = Image.open('images/fishingboat.tiff')
img = np.array(img,dtype=np.uint8)
freq = np.fft.fft2(img,s=(2*img.shape[0],2*img.shape[1]))
freqS = np.fft.fftshift(freq)

gausKernel = gaussian(20,3)
#gausKernel = np.fft.fft2(gausKernel)
gausKernelS = np.fft.fftshift(gausKernel)

dirac = dirac(20)
#dirac = np.fft.fft2(dirac)
diracS = np.fft.fftshift(dirac)

kernel = dirac+1*(dirac-gausKernel)
kernel = np.fft.fft2(kernel,s=(2*img.shape[0],2*img.shape[1]))
filtered = freq *  kernel
#filtered = np.fft.ifftshift(filtered)
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

plt.imshow(np.log2(1+abs(gausKernel)),cmap=plt.cm.gray)
plt.show()

plt.imshow(img2,cmap=plt.cm.gray,vmin = 0,vmax = 255)
plt.show()

