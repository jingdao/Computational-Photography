from math import *
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np

def apply_gaussian_filter(image, sigma, filterSize):
	new_image = image
	numRows = np.shape(image)[0]
	numCols = np.shape(image)[1]
	gFilter = calculate_gaussian_filter(sigma, filterSize)
	print gFilter
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


#Implementation of Laplacian pyramid
def laplacian_pyramid():
	pass

if __name__ == "__main__":
	sigma = 1 #MATLAB's default: 0.5
	filterSize = 5
	#calculate_gaussian_filter(sigma, filterSize)
	image = ndimage.imread('samples/mona-leber-target.jpg')
	imageRed = image[:,:,0]
	filtered_image = apply_gaussian_filter(imageRed,sigma,filterSize)
	plt.imshow(filtered_image)
	plt.show()