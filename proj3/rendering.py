import numpy as np

#take in array of frame indices from the original video
#crossfade where we make a different transition than original video:

#take in matrix of frames: each frame is a column
#average between current and next frame in the video texture
#at transition you should be averaging equally between current and next frame
def crossfade(frameIndices,video):
	for index in range(0,len(frameIndices) - 1):
		#we make a transition that did not happen in original video
		if frameIndices[index+1] != frameIndices[index] + 1:
			#weight scene before transition and scene after transition equally
			#equivalent to a weighting kernel of "radius" 1 (done for simplicity)
			video[frameIndices[index]] = 0.5*(video[frameIndices[index]] + video[frameIndices[index+1]])
	
	return video