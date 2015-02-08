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
l = 100.0 #smoothness


numSamples = 100 #sample numSamples random pixels to calculate response function
numImagesToUse = 8

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
	scaledMap = scalingValue * raw_map
	scaledMap[scaledMap > Zmax] = Zmax
	print "min value in scaled map: ", np.min(scaledMap)
	print "max value in scaled map: ", np.max(scaledMap)
	return scaledMap
	
#run the program
if __name__=="__main__":
	#height is number of rows, width is number of columns
	sampleRed,sampleGreen,sampleBlue,exposures,weights, imageRed, \
	imageGreen, imageBlue,numRowsInImage,numColsInImage \
	= getPixelArrayFromFiles('memorial','memorial.hdr_image_list.txt',numSamples)
	print "got pixel arrays"
	
	
	rfRed_sample,hdrMapRed_sample = rfsolve(sampleRed,exposures,l,weights,numImagesToUse)
	rfGreen_sample, hdrMapGreen_sample = rfsolve(sampleGreen,exposures,l,weights,numImagesToUse)
	rfBlue_sample, hdrMapBlue_sample = rfsolve(sampleBlue,exposures,l,weights,numImagesToUse)
	print "solved for response functions"

	hdrMapRed_images = create_map(rfRed_sample,imageRed,exposures,weights,numRowsInImage,numColsInImage,numImagesToUse)
	print "got first hdr map"
	hdrMapGreen_images = create_map(rfGreen_sample,imageGreen,exposures,weights,numRowsInImage,numColsInImage,numImagesToUse)
	print "got second hdr map"
	hdrMapBlue_images = create_map(rfBlue_sample,imageBlue,exposures,weights,numRowsInImage,numColsInImage,numImagesToUse)
	'''
	rfRed_images, hdrMapRed_images = rfsolve(imageRed,exposures,l,weights,numImagesToUse)
	print "got first hdr map"
	rfGreen_images, hdrMapGreen_images = rfsolve(imageGreen,exposures,l,weights,numImagesToUse)
	print "got second hdr map"
	rfBlue_images, hdrMapBlue_images = rfsolve(imageBlue,exposures,l,weights,numImagesToUse)
	'''
	
	print "got last hdr map, displaying based on hdr map of images"
	displayHDR(hdrMapRed_images,hdrMapGreen_images, hdrMapBlue_images)




	