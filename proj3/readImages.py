from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation

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
			pixelArray=imRed1D
		else:
			imRed1D=np.mat(imArr[:,:,0]).A1
			pixelArray=np.vstack((pixelArray,imRed1D))

	return pixelArray.T,numFrames,imHeight,imWidth

def animate(pixelArray,numFrames,imHeight,imWidth):
	ims=[]
	fig = plt.figure()
	for i in range(0,numFrames):
		imAxes = plt.imshow(pixelArray[:,0].reshape((imHeight,imWidth)),cmap=cm.Greys_r)
		ims.append([imAxes])
	matplotlib.animation.ArtistAnimation(fig,ims,interval=33,blit=True)
	plt.show()

if __name__=="__main__":
	pixelArray,numFrames,imHeight,imWidth = getPixelArrayFromFiles('clock')
	animate(pixelArray,numFrames,imHeight,imWidth)
