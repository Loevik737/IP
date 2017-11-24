%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import math as m
from PIL import Image
from scipy import fftpack


#implementation of von Neumann where we recursively searches for possible erighbors in the 4 adjacent cells
def segment(img,treshold,seed,searchSpot,retImg):
    row = seed[0]
    col = seed[1]
    srow = searchSpot[0]
    scol = searchSpot[1]
    new = []

    if srow-1 < 0 or srow+1 > img.shape[0]-1 or scol-1 <0 or scol +1 > img.shape[1]-1:
        return retImg
    if (abs(img[srow-1][scol] - img[row][col]) < treshold) and retImg[srow-1][scol] != 1:
        retImg[srow-1][scol] = 1
        new.append([srow-1,scol])

    if (abs(img[srow+1][scol] - img[row][col]) < treshold) and retImg[srow+1][scol] != 1:
        retImg[srow+1][scol] = 1
        new.append([srow+1,scol])

    if (abs(img[srow][scol-1] - img[row][col]) < treshold) and retImg[srow][scol-1] != 1:
        retImg[srow][scol-1] = 1
        new.append([srow,scol-1])

    if (abs(img[srow][scol+1] - img[row][col]) < treshold) and retImg[srow][scol+1] != 1:
        retImg[srow][scol+1] = 1
        new.append([srow,scol+1])

    if new != []:
        for pos in new:
            retImg = segment(img,treshold,seed,pos,retImg)
    return retImg


#for all the seeds find write 1 to the regions and 0 to everything else
def regionGrow(img,treshold,seeds):
    retImg = np.zeros(img.shape,dtype=np.uint8)
    for seed in seeds:
        retImg[seed[0]][seed[1]] = 1
        print("Seed:",img[seed[0]][seed[1]])
        searchSpot = seed
        retImg = segment(img,treshold,seed,searchSpot,retImg)
    return retImg

img = Image.open('images/Fig1043(a)(yeast_USC).tiff').convert('L')
img = np.array(img,dtype=np.uint8)

treshold = 40
seeds = [[700,590],[590,290]]
segImg = regionGrow(img,treshold,seeds)

plt.figure()
plt.imshow(img, cmap=plt.cm.gray)
plt.show()
plt.imshow(segImg,cmap=plt.cm.gray)
plt.show()
