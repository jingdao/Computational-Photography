#Based on Debevec and Malik (SIGGRAPH 1997), 
#"Recovering High Dynamic Range Radiance Maps from Photographs"

import numpy as np
import math as math

#minimum, maximum RGB color values
Zmin = 0
Zmax = 255

n = 256

#arguments:
#Z[i,j]: pixel values of of pixel location number i in image j
#B[j]: log delta t, or log shutter speed, for image j
#w[Z[i,j]]: weights for different pixel values
#g[Z[i,j]]: response function (vector of function values for all pixel values between Zmin and Zmax)

#returns:
#E(i) a HDR radiance map

def create_map(g,Z,B,w,numPixels,numImages):
	#High dynamic range radiance map
	#hdrMap is an array of with as many entries as there are pixels in any of our images
	#one entry corresponds to the HDR value of a given pixel in the radiance map
	hdrMap = list()
	
	#fill in each entry of the HDR radiance map
	for i in range(0,numPixels):
		num = 0
		denom = 0
		
		#use pixel values for a given pixel from all the images
		#difference between response function and exposure, weighted appropriately(?)
		for j in range(0,numImages):
			num += w[Z[i,j]]*(g[Z[i,j]] - B[j])
			denom += w[Z[i,j]]
		
		lnEi = num/denom
		hdrMap.append(math.exp(lnEi)) #do I need to append E_i or ln(E_i)?
	return hdrMap




#test	
numPixels = 16
numImages = 2

Z = np.zeros((numPixels,numImages))
g = np.ones(n)
w = np.ones(n)
B = np.ones(numImages)

print create_map(g,Z,B,w,numPixels,numImages)
#should be just ones, since each entry of the radiance map is e^0 = 1
	


	