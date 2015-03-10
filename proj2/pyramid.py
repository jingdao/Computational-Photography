from math import *
import numpy as np

def gaussian_pyramid(image, sigma, filterSize):
	gFilter = gaussian_filter(sigma, filterSize)
	#TODO use the filter to construct Gaussian pyramid

#Create square Gaussian filter of given size and sigma
#TODO: vectorize these steps instead of using for loops (may not matter much for small matrices like what we have here)
#TODO: (optional) allow for arbitrary rectangular Gaussian filters
def gaussian_filter(sigma, filterSize):
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
	gaussian_filter(sigma, filterSize)