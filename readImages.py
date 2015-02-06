#from scipy import misc
from scipy import ndimage
import numpy as np
import math
import matplotlib.pyplot as plt
from rfsolver import rfsolve

f=open('memorial/memorial.hdr_image_list.txt')
f.readline()
f.readline()
f.readline()
numImages=0
numPixels=0
B=[]
images=[]
x=np.array([])
y=np.array([])
z=[]
for s in f:
	sArray=s.split(' ')
	imFile=sArray[0].replace('.ppm','.png')
	exposureTime=math.log(1/float(sArray[1]))
	B.append(exposureTime)
	imArr=ndimage.imread('memorial/'+imFile)
	imRed=imArr[:,:,0]
	imSize=imRed.shape[0]*imRed.shape[1]
	imRed1D=np.mat(imRed).A1
	images.append(imRed)
	y=np.hstack((y,imRed1D))
	x=np.hstack((x,np.ones(imSize)*exposureTime))
	z.append(imRed1D)
	numImages+=1
	numPixels+=imSize

n = 256
B=np.array(B)
images=np.array(images)
z=np.transpose(np.array(z))
subSize=1000
z=z[0:subSize,:]
w=np.ones((n,1))
Zmin = 0
Zmax = 255
Zmid = (Zmin+Zmax)/2
for i in range(0,n):
	if i<=Zmid:
		w[i]=i-Zmin
	else:
		w[i]=Zmax-i
#print numImages,numPixels
#print images.shape
#print z.shape
#print z
#print B
out=rfsolve(z,B,10,w);
g=out[0]
xx=np.zeros(z.shape[0]*z.shape[1])
yy=np.zeros(z.shape[0]*z.shape[1])
k=0
for i in range(0,z.shape[0]):
	for j in range(0,z.shape[1]):
		xx[k]=g[z[i,j]]
		yy[k]=z[i,j]
		k=k+1
#print xx
#print yy
plt.plot(xx,yy,'x')
#plt.axis([0,300,-6,6])
plt.show()
f.close()
