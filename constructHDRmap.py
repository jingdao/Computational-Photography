#Based on Debevec and Malik (SIGGRAPH 1997), 
#"Recovering High Dynamic Range Radiance Maps from Photographs"
#we use this for calculating radiance maps for entire images
# using the response function we calculate based on a smaller subsample of the image

import numpy as np
import math as math

#minimum, maximum RGB color values
Zmin = 0
Zmax = 255

n = 256

#arguments:
#Z[i,j]: pixel values of of i,j-th pixel location number in a given RGB channel
#B[j]: log delta t, or log shutter speed, for image j
#w[Z[i,j]]: weights for different pixel values
#g[Z[i,j]]: response function (vector of function values for all pixel values between Zmin and Zmax)

#returns:
#E(i) a HDR radiance map for the given RGB channel of the image
#(call this on all three channels to get HDR radiance map for entire image in all color channels)
#we work with one channel at a time, as in the paper

def create_map(rfunc,image,exposure,weights):
	numRows = imageChannel.shape[0]
	numColumns = imageChannel.shape[1]
	#High dynamic range radiance map
	#hdrMap is an array of with as many entries as there are pixels in any of our images
	#one entry corresponds to the HDR value of a given pixel in the radiance map
	hdrMap = list()
	
	#fill in each entry of the HDR radiance map
	for i in range(0,numRows):
		for j in range(0,numColumns):
			num = 0
			denom = 0
			
			num += weights[imageChannel[i,j]]*([rfuncimageChannel[i,j]] - exposure)
			denom += weights[imageChannel[i,j]]
		
		lnEi = num/denom
		hdrMap.append(math.exp(lnEi))
	hdrMap = np.reshape(hdrMap, (numRows, numColumns))
	return hdrMap
	


	