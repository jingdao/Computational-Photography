from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.sparse.linalg



#source: region from source image to place in target image
#target: image into which to place the source region
#mask: specifies which part of the source image to place in the target image
	#matrix: 1 if corresponding pixel is in region to be put in target, 0 otherwise

#alignSource.m says where in the target image to put the region from source image

#TODO: separate into 3 color channels
def poissonBlend(source, target, mask):
	#Step 1: select source region and place to put it in target image, align source and target image
	#can manually do in Matlab for now
	#TODO: do in Python
	
	#Step 2: solve blending constraints
	#TODO: set source image (and region) and target image (and region) later
	sourceWidth = source.shape[1]
	sourceHeight = source.shape[0]
	
	targetWidth = target.shape[1]
	targetHeight = target.shape[0] 
	
	#set up least squares problem: Ax = b solving for x
	#keep track of indices of nonnegative entries in what will be a sparse matrix
	ArowIndices = []
	AcolIndices = []
	Adata = [] #list of equations for the vector A
	b = [] #list of equations for the vector b
	#later we can make these into sparse matrices
	
	#keep track of number of equations
	e = 0
	
	#EXPLORE: assign weighting factor for gradient constraint vs value constraint
	#(instead of just 1s and -1's: equal weight)
	
	#EXPLORE: size of neighborhood: (4 pixel neighborhood versus 8 pixel neighborhood)

	#for each pixel in the source image
	for i in range(0,sourceWidth): #column
		for j in range(0,sourceHeight): #row
			if mask[j,i]: #this pixel is in the source region
				originalPixelValue = source[j,i]
				#for each neighboring pixel in the source *that we haven't checked*
				#don't check neighbors you've already checked 
				#(e.g. comparing pixels [1,1] and [2,1] is same as comparing pixels [2,1] and [1,1])
				for m in range(0,2):
					for n in range(0,2):
						#two conditions:
						#either m or n is nonzero, so pixel's neighbor is distinct from it
						#neighboring pixel is also in the source region
						if m + n > 0 and mask[j+n,i+m]:
							ArowIndices.append(e) #the e-th equation
							AcolIndices.append(j*sourceWidth + i) #which entry
							Adata.append(1) #v_i
						
							neighbor = source[j+n,i+m]
							ArowIndices.append(e)
							AcolIndices.append((j+n)*sourceWidth + (i+m))
							Adata.append(-1) #-v_j
						
							#note: make the solution be a float
							b.append(float(neighbor) - originalPixelValue)
							e += 1
						if m + n > 0 and not mask[j+n,i+m]:
							ArowIndices.append(e) #the e-th equation
							AcolIndices.append(j*sourceWidth + i) #which entry
							Adata.append(1) #v_i
						
							neighbor = source[j+n,i+m]
						
							#target image pixel value is a constant with respect to x, what we're solving for
							#note: make the solution be a float
							b.append(float(neighbor) - originalPixelValue + target[(j+n),(i+m)])
							e += 1
						
	#shape of the matrix is (number of equations) x (number of pixels in source/target image)
	A=scipy.sparse.csr_matrix((Adata,(ArowIndices,AcolIndices)),shape=(e,sourceWidth * sourceHeight))

	#solve the least squares problem
#	x = np.linalg.lstsq(A,b)[0]
	b = np.array(b)
	x = scipy.sparse.linalg.lsqr(A,b)[0]
	x = x.reshape(source.shape)
	
	print("x, the source region: ")
	print x
	print("A, the equations: ")
	print A
	print("b, the solutions: ")
	print b	
	
	#return the solution: pixels of x corresponding to the source region should now be blended
	return x	

if __name__ == "__main__":
	source = np.matrix('1 2 3; 4 5 6; 7 8 9')
	target = np.matrix('3 1 8; 2 4 4; 7 9 5')
	mask = np.matrix('1 1 0; 1 1 0; 0 0 0')
	poissonBlend(source, target, mask)