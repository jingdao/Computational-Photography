import numpy as np

#takes in two vectors which we assume are of the same length
#computes l2 distance between them
def l2(image1, image2):
	#differences between each entry in the vector
	differences = np.subtract(image1, image2)
	
	#square the differences and sum them
	squaredDifferences = np.sum(np.square(differences))
	
	#l2 distance is square root of sum of squared differences
	return np.sqrt(squaredDifferences)


#map an l2 distance to a probability (number between 0 and 1) for transition
#sigma is a hyperparameter controlling the probability of a transition
#smaller sigmas are more conservative (only the best transitions are permitted)
#usage: determine probability of transition from frame1 to frame2
#determine a probability based on the distance between frame1's successor and frame2
def dist2prob(distance, sigma):
	return np.exp(-distance/sigma)
	
#get matrix of L2 distances between all frames in the video
def differencesMatrix(frames):
	numFrames = np.shape(frames)[1]
	diffMatrix = np.zeros(numFrames,numFrames)
	for frame1 in range(0,numFrames):
		for frame2 in range(0,numFrames):
			#note: if frames are the same, distance will be zero, as it already is
			#we know that distance from frame2 to frame1 is same as from frame1 to frame2
			#so without loss of generality assume frame1 < frame2 and cover the case where
			#frame1 > frame2 by symmetry
			if frame1 < frame2:
				dist = l2(frame1,frame2)
				diffMatrix[frame1,frame2] = dist
				diffMatrix[frame2,frame1] = dist
	return diffMatrix

#filter the difference matrix, so as to match subsequences instead of individual frames	
def filter(diffMatrix):
	numFrames = np.shape(frames)[1]
	filteredDists = np.zeros(numFrames,numFrames)
	#binomial weights
	#ideally calculate based on m (in the paper's notation), but in practice m = 1 or 2
	m = 2
	weights = [0.125,0.375,0.375,0.125]
	for frame1 in range(0,numFrames):
		for frame2 in range(0,numFrames):
			filteredVal = 0
			for index in range(0,len(weights)):
				filteredVal += diffMatrix[frame1+index-m, frame2+index-m]*weights[index]
			filteredDists[frame1,frame2] = fileredVal
	return filteredDists

#get matrix of probabilities of transitions between frames
def probabilityMatrix(distanceMatrix, sigma):
	numFrames = np.shape(frames)[1]
	probMatrix = np.zeros(numFrames,numFrames)
	for frame1 in range(0,numFrames):
		#so successor of last frame is the first frame
		successor = (frame1 + 1) % numFrames
		for frame2 in range(0,numFrames):
			distFromSuccessor = distanceMatrix[successor,frame2]
			probMatrix[frame1,frame2] = dist2prob(distFromSuccessor, sigma)
	#normalize rows in probability matrix
	#sums of rows by taking dot product with column vector of ones
	probMatrix = probMatrix/probMatrix.sum(axis=1,keepdims = True)
	return probMatrix
		
def predictFutureCost():
	pass	