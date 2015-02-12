#Based on Debevec and Malik (SIGGRAPH 1997), 
#"Recovering High Dynamic Range Radiance Maps from Photographs"
from scipy import ndimage
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors

#estimates noise in the given RGB channel of a given image (2D matrix)
#take random sample of pixels
#for each pixel in the image, see how different it is from its neighbors
def estimateNoise(imageChannelMatrix, numSamples):
	numRows = imageChannelMatrix.shape[0]
	numCols = imageChannelMatrix.shape[1]
	pixelNoise = []
	numNeighbors = 8 #every (interior) pixel always has 8 neighbors
	
	while len(pixelNoise) < numSamples:
		#only sample from the interior (non-edge pixels) of the image
		#that's probably more reliable anyway
		#and we don't have to worry about out of bounds errors
		randomRow = random.randint(1,numRows - 2)
		randomCol = random.randint(1,numCols - 2)
		
		pixelVal = int(imageChannelMatrix[randomRow, randomCol])
		sum_NeighborSquaredDifferences = 0
		for i in range(randomRow - 1, randomRow + 2):
			for j in range(randomCol - 1, randomCol + 2):
				if i != 0 or j != 0: #we're not comparing our pixel to itself
					#append squared difference of the pixel's color with that of its neighbor
					sum_NeighborSquaredDifferences += (int(imageChannelMatrix[i,j]) - pixelVal)**2
		#compute average of the squares of the difference in pixel values between the pixel we sampled and its neighbors
		avg_NeighborSquaredDifferences = sum_NeighborSquaredDifferences / float(numNeighbors)
		pixelNoise.append(avg_NeighborSquaredDifferences)
	
	#return mean of average squared differences with neighbor over our sample
	return sum(pixelNoise) / float(len(pixelNoise))
