from scipy import ndimage, stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation

#use probability matrix to make new video
#probabilities: probability matrix of transitioning to other frames from the current one
def makeVideo(pixelArray,numFrames,imHeight,imWidth, probDists):
	videoPauseLength = 0.033
	plt.show()
	#TODO: get color information
	imAxes = plt.imshow(pixelArray[:,0].reshape((imHeight,imWidth)),cmap=cm.Greys_r)
	nextImage = 0 #start at 0th image
	while(True):
		#TODO: do reshaping up front
		nextImage = probDists[nextImage].rvs()
		imAxes.set_data(pixelArray[:,nextImage].reshape((imHeight,imWidth)))
		plt.pause(videoPauseLength)
	