#Based on Debevec and Malik (SIGGRAPH 1997), 
#"Recovering High Dynamic Range Radiance Maps from Photographs"
import numpy as np
from scipy import ndimage
import matplotlib as plt
import math
import cProfile, pstats, StringIO

from readImages import *
from constructHDRmap import *
from rfsolver import *

l = 100.0 #smoothness
defaultImageHeight = 1000
defaultImageWidth = 1000


numSamples = 100 #sample numSamples random pixels to calculate response function
numImagesToUse = 8

#display the HDR map with different options	
def displayHDR(mapRed,mapGreen,mapBlue):	
	scaledRed = scaleHDR(mapRed, 10**-3)
	plt.imshow(scaledRed)
	plt.figure()
	scaledGreen = scaleHDR(mapGreen,10**-3)
	plt.imshow(scaledGreen)
	plt.figure()
	scaledBlue = scaleHDR(mapBlue,10**-3)
	plt.imshow(scaledBlue)
	plt.figure()
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
	
#set lambda, the smoothness parameter, automatically based on how much noise there is
#which we estimate by the variance in pixel radiances in our sample

#pixelSample is an array of pixels where the i-th entry is the radiance of pixel i
def smoothness(meanNoiseValue,imageHeight,imageWidth):
	#penalize larger images as being more noisy
	#a) because our estimate of noise probably under corrects (if the image is larger, the pixels values
	#probably won't change as quickly so neighboring pixels are more likely to be similar in value)
	#b) because larger images tend to be noisier... 
	noiseEstimate = meanNoiseValue * (imageHeight/float(defaultImageHeight)) * (imageWidth/float(defaultImageWidth))
	
	#function converting noise estimate to a smoothness parameter
	#could do linear regression or something to get exact parameters, but that would require
	#mathematically evaluating how good your image is for a given smoothness parameter
	#so we just estimate it roughly
	smoothness = 0.1*noiseEstimate
	print("We estimate the noise of these images to be %f.  Setting lamdba to be %f" % (noiseEstimate, smoothness))
	return smoothness
	
#run the program
if __name__=="__main__":
	#start profiler
	pr = cProfile.Profile()
	pr.enable()
	
	#height is number of rows, width is number of columns
	sampleRed,sampleGreen,sampleBlue,exposures,weights, imageRed, \
	imageGreen, imageBlue,numRowsInImage,numColsInImage, meanNoiseValue \
	= getPixelArrayFromFiles('memorial','memorial.hdr_image_list.txt',numSamples)
	print "got pixel arrays"
	l = smoothness(meanNoiseValue,numRowsInImage,numColsInImage)
	
	
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
	#end profiler
	pr.disable()
	s = StringIO.StringIO()
	sortby = "cumtime"
	ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
	ps.print_stats()
	print s.getvalue()
	displayHDR(hdrMapRed_images,hdrMapGreen_images, hdrMapBlue_images)
	print "size of response function for red channel: ", rfRed_sample.size
	
	#plot the response functions
	x = np.zeros(rfRed_sample.size)
	for i in range(0,x.size):
		x[i] = i
	plt.plot(x,rfRed_sample, "r-", x,rfGreen_sample, "g-", x,rfBlue_sample, "b-")
	plt.show()




	