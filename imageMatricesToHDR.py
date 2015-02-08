#Based on Debevec and Malik (SIGGRAPH 1997), 
#"Recovering High Dynamic Range Radiance Maps from Photographs"
import numpy as np
from scipy import ndimage
import matplotlib as plt
import math
from readImages import *
from constructHDRmap import *
from rfsolver import *

#minimum, maximum RGB color values
Zmin = 0
Zmax = 255
Zmid = (Zmax + Zmin)/2

n = 256
l = 1.0 #smoothness


numSamples = 1000 #sample numSamples random pixels to calculate response function
numImagesToUse = 2

'''
#take in images one RGB channel at a time, get response function
#take response function and images, get HDR map
#now trimmed down so it basically just calls rfsolve...
def imageMatricesToHDR(images, exposures, smoothness, weights):
	print "solving for the response function and radiance map"
	rfMap = rfsolve(images,exposures,smoothness, weights)
	g = rfMap[0] #response function
	hdrmap = np.exp(rfMap[1]) #radiance map (of log exposures, so take exponential to get exposures)
	return g, hdrmap
'''

#display the HDR map with different options	
def displayHDR(mapRed,mapGreen,mapBlue):	
	scaledRed = scaleHDR(mapRed, 10**-3)
	scaledGreen = scaleHDR(mapGreen,10**-3)
	scaledBlue = scaleHDR(mapBlue,10**-3)
	combinedMap = np.dstack([scaledRed,scaledGreen,scaledBlue])
	print "showing image: "
	plt.imshow(combinedMap)
	plt.show()
	
#scaling as per option (c) in the paper
#everything above 0.1% of the highest value gets mapped to 255
#if x is the scale factor of the highest value, multiply x by 255/x so that x gets mapped to 255 as well
#multiply everything else by 255/x
def scaleHDR(raw_map, scale):
	maxValue = np.max(raw_map)
	print "max value in raw map: ", np.max(raw_map)
	print "min value in raw map: ", np.min(raw_map)
	scaledMax = scale * maxValue
	scalingValue = Zmax/scaledMax
	print "how large the min is, when scaled: ", np.min(raw_map)*scalingValue
	scaledMap = scalingValue * raw_map
	scaledMap[scaledMap > Zmax] = Zmax
	return scaledMap
	
#run the program
if __name__=="__main__":
	#height is number of rows, width is number of columns
	imagesRed,imagesGreen,imagesBlue,exposures,weights, finalImageRed, \
	finalImageGreen, finalImageBlue,numRowsInImage,numColsInImage \
	= getPixelArrayFromFiles('memorial','memorial.hdr_image_list.txt',numSamples)
	print "got pixel arrays"
	
	rfRed_sample,hdrMapRed_sample = rfsolve(imagesRed,exposures,l,weights)
	rfGreen_sample, hdrMapGreen_sample = rfsolve(imagesGreen,exposures,l,weights)
	rfBlue_sample, hdrMapBlue_sample = rfsolve(imagesBlue,exposures,l,weights)
	print "solved for response functions"

	hdrMapRed_images = create_map(rfRed_sample,finalImageRed,exposures,weights,numRowsInImage,numColsInImage,numImagesToUse)
	print "got first hdr map"
	hdrMapGreen_images = create_map(rfGreen_sample,finalImageGreen,exposures,weights,numRowsInImage,numColsInImage,numImagesToUse)
	hdrMapBlue_images = create_map(rfBlue_sample,finalImageBlue,exposures,weights,numRowsInImage,numColsInImage,numImagesToUse)

	print "displaying based on hdr map of images"
	displayHDR(hdrMapRed_images,hdrMapGreen_images, hdrMapBlue_images)




	