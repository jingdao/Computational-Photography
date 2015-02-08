#Based on Debevec and Malik (SIGGRAPH 1997), 
#"Recovering High Dynamic Range Radiance Maps from Photographs"
import numpy as np
from scipy import ndimage
import matplotlib as plt
import math
from constructHDRmap import *
from rfsolver import *

#minimum, maximum RGB color values
Zmin = 0
Zmax = 255
Zmid = (Zmax + Zmin)/2

n = 256

#RGB channel numbers
RED = 0
GREEN = 1
BLUE = 2

l = 1.0 #smoothness


imageSize = 25 #sample imageSizeximageSize piece of image for testing

#read in actual input (images and exposure times into an array)
#TODO perhaps: make it read in an arbitrary filename?
def readInput():
	#for now, we're working with a specific file with a specific format
	f = open('memorial/memorial.hdr_image_list.txt')
	#get rid of blank lines at the top
	
	f.readline()
	f.readline()
	f.readline()
	numImages=0 #keep counts of number of images, for testing
	numPixels=0 #keep count of number of pixels, for testing
	
	images = [] #holds matrices of images
	exposures=[] #keep track of (the log of) the exposures of each image
	for s in f:
		sArray=s.split(' ')
		
		imFile=sArray[0].replace('.ppm','.png') #read in exposure value
		exposureTime=math.log(1/float(sArray[1])) #calculate the log exposure value to be stored
		exposures.append(exposureTime) #store the log exposure for the image
		
		imArr=ndimage.imread('memorial/'+imFile) #read in an image
		images.append(imArr) #store the image we read
		
		numImages+=1
		numPixels+=imArr.shape[0]*imArr.shape[1]
	f.close()
	
	return images, exposures


#take in images, get response function
#take response function and images, get HDR map
def imageMatricesToHDR(images, exposures, smoothness, weights, channel):
	numImages = len(images)
	
	print "reading image 1 "
	firstImage = images[0][0:imageSize,0:imageSize,:] #by assumption, all images have same size
	numPixels = np.size(firstImage[:,:,channel])
	
	#create array Z where the columns are images and the rows are values for each pixel in an image
	#TODO: do this correctly
	
	#flatten 2D image into 1D vector of pixel values
	Z = np.matrix(np.ndarray.flatten(firstImage[:,:,channel]))

	#for the remainder of the images
	for i in range(1,numImages):
		print "reading image ", i + 1
		#concatenate flattened images vertically (because you can concatenate arrays vertically)
		newImage = images[i][0:imageSize,0:imageSize,:]
		Z = np.vstack((Z, np.ndarray.flatten(newImage[:,:,channel])) )
	
	#so that columns are images, per how the response function calculator accepts input	
	Z = Z.transpose()
	
	print "solving for the response function and radiance map"
	rfMap = rfsolve(Z,exposures,smoothness, weights)
	g = rfMap[0] #response function
	hdrmap = np.exp(rfMap[1]) #radiance map (of log exposures, so take exponential to get exposures)
	
	#reshape map to have same shape as one of the images instead of a very long array
	hdrmap = np.reshape(hdrmap, np.shape(firstImage[:,:,channel]))
	return hdrmap

#display the HDR map with different options	
def displayHDR(mapRed,mapGreen,mapBlue):	
	scaledRed = scaleHDR(mapRed, 10**-3)
	#scaledGreen = scaleHDR(mapGreen,10**-3)
	#scaledBlue = scaleHDR(mapBlue,10**-3)
	
#everything above 0.1% of the highest value gets mapped to 255
#if x is the scale factor of the highest value, multiply x by 255/x so that x gets mapped to 255 as well
#multiply everything else by 255/x
def scaleHDR(raw_map, scale):
	maxValue = np.max(raw_map)
	print "max value in raw map: ", np.max(raw_map)
	print "min value in raw map: ", np.min(raw_map)
	scaledMax = scale * maxValue
	scalingValue = Zmax/scaledMax
	print "scaling value: ", scalingValue
	print "how large the min is, when scaled: ", np.min(raw_map)*scalingValue
	scaledMap = scalingValue * raw_map
	scaledMap[scaledMap > Zmax] = Zmax
	return scaledMap
	

#run the program	
inputImagesExposures = readInput()

images = inputImagesExposures[0]
exposures = inputImagesExposures[1]

#TODO: this code is also in rfsolver.  remove one of them
weights = np.ones((n,1)) #calculate weights (as done in the paper)
for i in range(0,n):
	if i <= Zmid:
		weights[i] = i-Zmin
	else:
		weights[i] = Zmax-i

#gets HDR map for our input, using the red channel (channel 0)
#green channel (channel 1)
#blue channel (channel 2)
hdrMapRed = imageMatricesToHDR(images,exposures,l,weights,RED)
hdrMapGreen = imageMatricesToHDR(images,exposures,l,weights,GREEN)
hdrMapBlue = imageMatricesToHDR(images,exposures,l,weights,BLUE)
displayHDR(hdrMapRed,hdrMapGreen, hdrMapBlue)
	
	



	