from scipy import ndimage, stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation

#use probability matrix to make new video
#probabilities: probability matrix of transitioning to other frames from the current one
def makeVideo(pixelArrayRed,pixelArrayGreen,pixelArrayBlue,numFrames,imHeight,imWidth,fps,probDists):
	videoPauseLength = 1.0/fps
	frames=[]
	for i in range(0,numFrames):
		vRed=pixelArrayRed[:,i].reshape((imHeight,imWidth))
		vGreen=pixelArrayGreen[:,i].reshape((imHeight,imWidth))
		vBlue=pixelArrayBlue[:,i].reshape((imHeight,imWidth))
		frames.append(np.dstack((vRed,vGreen,vBlue)))
	plt.show()
	imAxes = plt.imshow(frames[0])
	nextImage = 0 #start at 0th image
	while(True):
#		imAxes.set_label('Frame index:'+str(nextImage))
		plt.title('Frame index:'+str(nextImage))
		nextImage = probDists[nextImage].rvs()
		imAxes.set_data(frames[nextImage])
		plt.pause(videoPauseLength)
	
