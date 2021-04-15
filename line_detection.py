#!python3

import cv2
import numpy as np
import matplotlib.pyplot as plt

def convolution(im, kern):
    kDim = (len(kern), len(kern[0]))
    assert kDim[0] % 2 == 1 and kDim[1] % 2 == 1, "Kernel must be odd size"
    kCen = (kDim[0]//2,kDim[1]//2)
    cDim = (len(im),len(im[0]))
    conv = np.zeros(cDim, dtype=np.int16)

    for r in range(cDim[0]):
        for c in range(cDim[1]):
            v = 0.0
            for kr in range(kDim[0]):
                for kc in range(kDim[1]):
                    try:
                        v = v + im[r+kr-kCen[0],c+kc-kCen[1]] * kern[kr,kc]
                    except:
                        pass
            conv[r][c] = v
    return conv

def doOp(im,op):
    cDim = (len(im),len(im[0]))
    newim = np.zeros(cDim, dtype=np.int16)
    
    for r in range(cDim[0]):
        for c in range(cDim[1]):
            newim[r][c] = op(im[r][c])
    return newim

def hist(im,bins):
    L = np.zeros((bins,1),dtype=np.float32)
    for i in im.ravel():
        L[i] +=1
    return L

def segment(im):
    h = hist(im,256)
    b = np.zeros((256,1),dtype=np.float32)
    for i in range(256):
        b[i] = i

    T = int(np.average(im))
    m1 = int(np.sum(h[:T]*b[:T])/np.sum(h[:T]))
    m2 = int(np.sum(h[T:]*b[T:])/np.sum(h[T:]))
    while(int((m1+m2)/2) !=T):
        T = int((m1+m2)/2)
        m1 = int(np.sum(h[:T]*b[:T])/np.sum(h[:T]))
        m2 = int(np.sum(h[T:]*b[T:])/np.sum(h[T:]))
    for r in range(len(im)):
        for c in range(len(im[0])):
            im[r,c] = 0 if im[r,c]<T else 255
    return (im,T)

def hough(im):
    dim = im.shape
    maxd = np.linalg.norm(dim)
    ts = np.arange(-np.pi,np.pi,np.pi/4)

    coss = np.cos(t)
    sins = np.sin(t)
    xs = [range(len(dim[1]))]

    ht = np.zeros((2*maxd,len(t)),dtype=np.float16)


im = cv2.imread("lines.png",cv2.IMREAD_GRAYSCALE)

sx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sy = sx.T

xconv = doOp(convolution(im,sx),abs)
xconv = xconv/np.max(xconv)
yconv = doOp(convolution(im,sy),abs)
yconv = yconv/np.max(yconv)
# cv2.imshow("X",xconv)
# cv2.imshow("Y",yconv)
# cv2.waitKey(0)

edges = xconv + yconv
edges = edges/np.max(edges)
# cv2.imshow("edges",edges)
# cv2.waitKey(0)

sim = segment(edges)
cv2.imshow("Segmented", sim)
cv2.waitKey(0)
him = hough(sim)