#Based on Debevec and Malik (SIGGRAPH 1997), 
#"Recovering High Dynamic Range Radiance Maps from Photographs"

import numpy as np
from constructHDRmap import *
from rfsolver import *

#minimum, maximum RGB color values
Zmin = 0
Zmax = 255

n = 256

#main method of this paper implementation
#take in images, get response function
#take response function and images, get HDR map
def main(images, exposures, smoothness, weights):
	numImages = len(images)
	
	firstImage = images[0] #by assumption, all images have same size
	numPixels = np.size(firstImage)
	
	#create array Z where the columns are images and the rows are values for each pixel in an image
	#TODO: do this correctly
	
	#flatten 2D image into 1D vector of pixel values
	Z = np.matrix(np.ndarray.flatten(firstImage))

	#for the remainder of the images
	for i in range(1,numImages):
		#concatenate flattened images vertically (because you can concatenate arrays vertically)
		Z = np.vstack((Z, np.ndarray.flatten(images[i])) )
	
	#so that columns are images, per how the response function calculator accepts input	
	Z = Z.transpose()
	
	g = rfsolve(Z,exposures,smoothness, weights)[0]
	hdrmap = create_map(g,Z,exposures,weights,numPixels,numImages)
	
	return hdrmap
	
#TO BE DONE: rewrite rfsolver so that it takes in only matrices, not functions
images = list()
for j in range(0,2):
	images.append( np.zeros((4,4)) )
exposures = np.ones(len(images))
l = 1.0
weights = np.ones(n)

sample_hdrMap = main(images,exposures,l,weights)

#reshape map to have same shape as one of the images instead of a very long array
#arbitrarily we make it same shape as first image (all images have same shape)
sample_hdrMap = np.reshape(sample_hdrMap, np.shape(images[0]))
print "Sample HDR radiance map has shape ", np.shape(sample_hdrMap)
	
	



	