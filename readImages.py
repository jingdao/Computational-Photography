#from scipy import misc
from scipy import ndimage
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from rfsolver import rfsolve

def getPixelArrayFromFiles(dirName,txtFile):
	f=open(dirName+'/'+txtFile)
	f.readline()
	f.readline()
	f.readline()
	numImages=0
	numPixels=0
	B=[]
	zRed=[]
	zGreen=[]
	zBlue=[]
	numSamples=10
	pixelSamples=set()
	for s in f:
		sArray=s.split(' ')
		imFile=sArray[0].replace('.ppm','.png')
		exposureTime=math.log(1/float(sArray[1]))
		B.append(exposureTime)
		imArr=ndimage.imread(dirName+'/'+imFile)
		imRed=imArr[:,:,0]
		imGreen=imArr[:,:,1]
		imBlue=imArr[:,:,2]
		imSize=imRed.shape[0]*imRed.shape[1]
		if len(pixelSamples)==0:
			while len(pixelSamples)<numSamples:
				i=random.randint(0,imSize-1)
				pixelSamples.add(i)
		imRed1D=[]
		imGreen1D=[]
		imBlue1D=[]
		for i in pixelSamples:
			imRed1D.append(imRed[i%imRed.shape[0],i/imRed.shape[0]])
			imGreen1D.append(imGreen[i%imGreen.shape[0],i/imGreen.shape[0]])
			imBlue1D.append(imBlue[i%imBlue.shape[0],i/imBlue.shape[0]])
		zRed.append(imRed1D)
		zGreen.append(imGreen1D)
		zBlue.append(imBlue1D)
		numImages+=1
		numPixels+=numSamples
	f.close()
	n = 256
	B=np.array(B)
	zRed=np.transpose(np.array(zRed))
	zGreen=np.transpose(np.array(zGreen))
	zBlue=np.transpose(np.array(zBlue))
	w=np.ones((n,1))
	Zmin = 0
	Zmax = 255
	Zmid = (Zmin+Zmax)/2
	for i in range(0,n):
		if i<=Zmid:
			w[i]=i-Zmin
		else:
			w[i]=Zmax-i
	return zRed,zGreen,zBlue,B,w

def plotZandG(z,g,color):
	xx=np.zeros(z.shape[0]*z.shape[1])
	yy=np.zeros(z.shape[0]*z.shape[1])
	k=0
	for i in range(0,z.shape[0]):
		for j in range(0,z.shape[1]):
			xx[k]=g[z[i,j]]
			yy[k]=z[i,j]
			k=k+1
	plt.plot(xx,yy,color)


if __name__=="__main__":
	zRed,zGreen,zBlue,B,w = getPixelArrayFromFiles('memorial','memorial.hdr_image_list.txt')
	l=1
	plotZandG(zRed,rfsolve(zRed,B,l,w)[0],'rx')
	plotZandG(zGreen,rfsolve(zGreen,B,l,w)[0],'gx')
	plotZandG(zBlue,rfsolve(zBlue,B,l,w)[0],'bx')
	plt.axis([-10,5,0,260])
	plt.show()
