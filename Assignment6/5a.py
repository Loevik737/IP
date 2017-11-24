%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import math as m
from PIL import Image
from scipy.ndimage.morphology import binary_erosion, binary_dilation,binary_fill_holes
from scipy import ndimage
from skimage import feature,exposure
from skimage.transform import hough_line,probabilistic_hough_line
from skimage.measure import approximate_polygon


def setLabel(img,seed,searchSpot,label):
    row = seed[0]
    col = seed[1]
    srow = searchSpot[0]
    scol = searchSpot[1]
    new = []
    if srow-1 < 0 or srow+1 > img.shape[0]-1 or scol-1 <0 or scol +1 > img.shape[1]-1:
        return img

    if img[srow-1][scol] == 1 and img[srow-1][scol] != label:
        img[srow-1][scol] = label
        new.append([srow-1,scol])

    if img[srow+1][scol] == 1 and img[srow+1][scol] != label:
        img[srow+1][scol] = label
        new.append([srow+1,scol])

    if img[srow][scol-1] == 1 and img[srow][scol-1] != label:
        img[srow][scol-1] = label
        new.append([srow,scol-1])

    if img[srow][scol+1] == 1 and img[srow][scol+1] != label:
        img[srow][scol+1] = label
        new.append([srow,scol+1])

    if new != []:
        for pos in new:
            img = setLabel(img,seed,pos,label)
    return img


def binImg(img):
    retImg = np.zeros(img.shape,dtype = np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j]:
                retImg[i][j] = 1
    return retImg

def invert(img):
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            img[i][j] = not img[i][j] 
    return img

def getLabels(img):
    consecutive = 0
    label = 2
    labels =[] 
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] == 1:
                if consecutive ==40:
                    img = setLabel(img,[i,j],[i,j],label)
                    labels.append(label)
                    consecutive = 0
                    label += 1
                else:
                    consecutive +=1
    return img,labels
            

#a
#open the image and convert it to grayscale.Then we do dilation to only have the checker board edges
imgFile = Image.open('images/task5-01.tiff')
orgImg = np.array(imgFile,dtype=np.uint8)
img = np.array(imgFile.convert('L'),dtype=np.uint8)
img = feature.canny(img,0.1)
img = binary_dilation(img)

#remove the checker board edges by makingthem bigger, inverting them and dot multiply them with the original edge image
#we are now left with only the shapes an som minor noice
vKernel = np.ones((30,1))
hKernel = np.ones((1,30))
img2 = binary_dilation(binary_erosion(img,structure=vKernel),structure=vKernel,iterations=15).astype(img.dtype)
img3 = binary_dilation(binary_erosion(img,structure=hKernel),structure=hKernel,iterations=15).astype(img.dtype)
img = img * (invert(img2) * invert(img3))

#some noice tweeking and filling the insides of the shapes with binary_fill_holes
img = binary_dilation(img)
img = binary_dilation(img,structure=[[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]],iterations=2) 
img = binary_fill_holes(img)
#this tweaking is only to make 5-02 work
img = binary_erosion(img,iterations=10)
img=binImg(img)
binimg = binImg(img)

#b
#give everithing of a significant size a label
labelImg,labels = getLabels(img)

#c
#find  the center of mass of the labeled shapes
center = ndimage.measurements.center_of_mass(img,labelImg,labels)
centerX = []
centerY = []
for a,b in center:
    centerY.append(int(round(a)))
    centerX.append(int(round(b)))


#a trie on recognicing shapes by counting lines and then counting the number of detected lines to know what shape it is by
#using the Hough transform lines function.

"""
for k in range(len(centerX)):
    x = centerX[k]
    y = centerY[k]
    testImg = labelImg[y-38:y+38,x-38:x+38]
    for i in range(testImg.shape[0]):
        for j in range(testImg.shape[1]):
            if testImg[i][j] != 0:
                testImg[i][j] = 255
    testImg = feature.canny(testImg)
    lines = probabilistic_hough_line(testImg,threshold = 10,line_gap= 3, line_length=40)
    print(k,':',lines)
    if len(lines) == 3:
        print("Triangle at",x,y)
    elif len(lines) == 4:
        print("Paralellogram at",x,y)
    elif len(lines) == 0:
        print("Circle at",x,y

    ...
    ...

"""
print(lines)
plt.figure()

plt.imshow(binimg,cmap=plt.cm.gray)
plt.show()

plt.imshow(orgImg,zorder=1)
plt.scatter(centerX,centerY,zorder=2)
plt.show()

plt.imshow(labelImg,cmap=plt.cm.gray)
plt.show()


