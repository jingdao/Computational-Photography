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
	plt.axis('off')
	nextImage = 0 #start at 0th image
	while(True):
		plt.title('Frame index:'+str(nextImage))
		nextImage = probDists[nextImage].rvs()
		imAxes.set_data(frames[nextImage])
		plt.pause(videoPauseLength)

#same function as above except this time the result is not displayed but saved to a video file
def saveVideo(pixelArrayRed,pixelArrayGreen,pixelArrayBlue,numFramesInitial,numFramesFinal,imHeight,imWidth,framesPerSecond,probDists,showIndex):
	fig = plt.figure()
	plt.axis('off')
	frames=[]
	for i in range(0,numFramesInitial):
		vRed=pixelArrayRed[:,i].reshape((imHeight,imWidth))
		vGreen=pixelArrayGreen[:,i].reshape((imHeight,imWidth))
		vBlue=pixelArrayBlue[:,i].reshape((imHeight,imWidth))
		frames.append(np.dstack((vRed,vGreen,vBlue)))
	if not showIndex:
		nextImage=0
		artists=[]
		for i in range(0,numFramesFinal):
			imAxes=plt.imshow(frames[nextImage])
			artists.append([imAxes])
			nextImage = probDists[nextImage].rvs()
		ani=matplotlib.animation.ArtistAnimation(fig,artists,interval=1000/framesPerSecond,blit=True)
	else:
		imAxes=plt.imshow(frames[0])
		ttl=plt.title('Frame index:0')
		def callback(i):
			callback.nextImage=probDists[callback.nextImage].rvs()
			imAxes.set_data(frames[callback.nextImage])
			ttl.set_text('Frame index:'+str(callback.nextImage))
		callback.nextImage=0
		ani=matplotlib.animation.FuncAnimation(fig,callback,np.arange(0,numFramesFinal),interval=1000/framesPerSecond,blit=True)
	ani.save('out.mp4',fps=framesPerSecond)
	
