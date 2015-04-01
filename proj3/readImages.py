from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation
from frameDistances import *

#	get pixel values from all frames of a video
#	dirName: directory where video frame images are stored
#	pixelArray: 2D array where each column is a frame
def getPixelArrayFromFiles(dirName):
	f = open(dirName+"/data.txt")
	numFrames = int(f.readline())
	fps = int(f.readline()) #frames per second
	f.close()

	for i in range(1,numFrames+1):
		imArr=ndimage.imread(dirName+'/'+str(i)+'.png')
		if i==1:
			imHeight=imArr.shape[0]
			imWidth=imArr.shape[1]
			imRed1D=np.mat(imArr[:,:,0]).A1
			pixelArrayRed=imRed1D
			imGreen1D=np.mat(imArr[:,:,1]).A1
			pixelArrayGreen=imGreen1D
			imBlue1D=np.mat(imArr[:,:,2]).A1
			pixelArrayBlue=imBlue1D
		else:
			imRed1D=np.mat(imArr[:,:,0]).A1
			pixelArrayRed=np.vstack((pixelArrayRed,imRed1D))
			imGreen1D=np.mat(imArr[:,:,1]).A1
			pixelArrayGreen=np.vstack((pixelArrayGreen,imGreen1D))
			imBlue1D=np.mat(imArr[:,:,2]).A1
			pixelArrayBlue=np.vstack((pixelArrayBlue,imBlue1D))

	return pixelArrayRed.T,pixelArrayGreen.T,pixelArrayBlue.T,numFrames,imHeight,imWidth

def animate(pixelArrayRed,pixelArrayGreen,pixelArrayBlue,numFrames,imHeight,imWidth):
	fig = plt.figure()
	frames=[]
	for i in range(0,numFrames):
		vRed=pixelArrayRed[:,i].reshape((imHeight,imWidth))
		vGreen=pixelArrayGreen[:,i].reshape((imHeight,imWidth))
		vBlue=pixelArrayBlue[:,i].reshape((imHeight,imWidth))
		imAxes=plt.imshow(np.dstack((vRed,vGreen,vBlue)))
		frames.append([imAxes])
	ani=matplotlib.animation.ArtistAnimation(fig,frames,interval=33,blit=True)
	plt.show()

if __name__=="__main__":
	pixelArrayRed,pixelArrayGreen,pixelArrayBlue,numFrames,imHeight,imWidth = getPixelArrayFromFiles('clock')
	animate(pixelArrayRed,pixelArrayGreen,pixelArrayBlue,numFrames,imHeight,imWidth)
#	diff = differencesMatrix(pixelArray)
#	prob = probabilityMatrix(pixelArray,diff,1000)
#	plt.figure()
#	plt.imshow(dist)
#	plt.colorbar()
#	plt.figure()
#	plt.imshow(prob)
#	plt.colorbar()
#	plt.show()

