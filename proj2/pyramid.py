from math import *
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np

#create a Gaussian pyramid from an image using a given sigma
#and a square Gaussian filter of a given odd size
#one color channel at a time
def createLaplacianPyramid(image,sigma,filterSize,numLayers):
	#represent pyramid as list of layers
	gPyramid = [image] #first layer of pyramid
	for i in range(1,numLayers):
		#convolve the previous layer with Gaussian filter
		nextLayer = apply_gaussian_filter(gPyramid[i-1],sigma,filterSize)
		#subsample
		nextLayer = subsample(nextLayer)
		gPyramid.append(nextLayer)
	return gPyramid

#subsample a matrix: get every other row and column
#TODO: vectorize if possible	
def subsample(matrix):
	numRows = np.shape(matrix)[0]
	numCols = np.shape(matrix)[1]
	
	#holds the values of the subsampled matrix
	newMatrix = np.zeros((numRows/2 + 1, numCols/2 + 1))
	
	#sample odd (by human counting--even by index) rows and columns
	for row in range(0,numRows):
		if row % 2 == 0:
			for col in range(0,numCols):
			 	if col % 2 == 0:
					newMatrix[row/2,col/2] = matrix[row,col]
	
	#return final result			
	return newMatrix
			

#apply a (square, of a given (odd) size) Gaussian filter to an image
#one color channel at a time
def apply_gaussian_filter(image, sigma, filterSize):
	new_image = image
	numRows = np.shape(image)[0]
	numCols = np.shape(image)[1]
	gFilter = calculate_gaussian_filter(sigma, filterSize)
	#print gFilter
	halfFilterSize = (filterSize - 1)/2
	#apply Gaussian filter where applicable (i.e. not to the edges)
	for row in range(halfFilterSize, numRows - halfFilterSize):
		for col in range(halfFilterSize, numCols - halfFilterSize):
			#take the pixels in a filterSize*filterSize square around current pixel (assume odd length side)
			neighborhood = image[row - halfFilterSize:row+halfFilterSize + 1,col - halfFilterSize:col+halfFilterSize + 1]
			#print neighborhood
			#print sum(np.multiply(neighborhood,gFilter))
			new_image[row,col] = np.sum(np.multiply(neighborhood,gFilter))
			
	return new_image
	#TODO use the filter to construct Gaussian pyramid

#Create square Gaussian filter of given size and sigma
#TODO: vectorize these steps instead of using for loops (may not matter much for small matrices like what we have here)
#TODO: (optional) allow for arbitrary rectangular Gaussian filters
def calculate_gaussian_filter(sigma, filterSize):
	m1 = np.zeros((filterSize, filterSize))
	gFilter = np.zeros((filterSize, filterSize))
	
	#create initial values
	for row in range(0,filterSize): #iterate over all rows of filter
		for entry in range(0,filterSize):
			#number of columns away from the center
			m1[row,entry] = int(entry - filterSize / 2)
	
	m1 = m1.astype(int)
	#number of rows away from the center
	m2 = m1.transpose()
			
	#convolute with Gaussian kernel
	for row in range(0,filterSize):
		for column in range(0,filterSize):
			gFilter[row,column] = gaussian_kernel(m1[row,column],m2[row,column],sigma)
	gFilter = gFilter / np.sum(gFilter)
	return gFilter
	
#Gaussian kernel
def gaussian_kernel(x1,x2,sigma):
	squaredDifference = (x1**2 + x2**2)
	return exp(-1.0*squaredDifference/(2*sigma**2))


#test
if __name__ == "__main__":
	sigma = 1 #MATLAB's default: 0.5
	filterSize = 5
	#calculate_gaussian_filter(sigma, filterSize)
	image = ndimage.imread('samples/mona-leber-target.jpg')
	imageRed = image[:,:,0]
	imageGreen = image[:,:,1]
	imageBlue = image[:,:,2]
	numLayers = 3
	pyramidRed = createLaplacianPyramid(imageRed,sigma,filterSize,numLayers)
	pyramidGreen = createLaplacianPyramid(imageGreen,sigma,filterSize,numLayers)
	pyramidBlue = createLaplacianPyramid(imageBlue,sigma,filterSize,numLayers)
	#display images in pyramid
	for i in range(0,numLayers):
		print("Layer %d has shape " % i)
		print np.shape(pyramidRed[i])
		
		#cast to ints
		pyramidRed[i] = np.array(pyramidRed[i],dtype=np.uint8)
		pyramidGreen[i] = np.array(pyramidGreen[i],dtype=np.uint8)
		pyramidBlue[i] = np.array(pyramidBlue[i],dtype=np.uint8)
		
		pyramidLayerImage = np.dstack([pyramidRed[i],pyramidGreen[i],pyramidBlue[i]])
		plt.figure()
		plt.imshow(pyramidLayerImage)
	plt.show()