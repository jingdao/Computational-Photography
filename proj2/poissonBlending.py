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
	sourceArr = ndimage.imread('samples/toy_problem.png')
	sourceWidth = sourceArr.shape[1]
	sourceHeight = sourceArr.shape[0]
	
	targetArr = ndimage.imread('samples/toy_problem.png')
	targetWidth = targetArr.shape[1]
	targetHeight = targetArr.shape[0]
	
	#keep track of number of equations
	e = 0
	
	#set up least squares problem: Ax = b solving for x
	A = list() #list of equations for the vector A
	b = list() #list of equations for the vector b
	#later we can make these into sparse matrices
	
	#for each pixel in the source image
	for i in range(0,sourceWidth):
		for j in range(0,sourceHeight):
			
			e += 1
			
	
	'''
	imArr=ndimage.imread('samples/toy_problem.png')
	imWidth=imArr.shape[1]
	imHeight=imArr.shape[0]
	numVariables=imWidth*imHeight
	numEquations=(imWidth-1)*imHeight+(imHeight-1)*imWidth+1
	numIndices=(numEquations-1)*2+1
#	A=np.zeros((numEquations,numVariables))
	#A_indices=np.zeros(numIndices,dtype=np.int64)
	A_indices 
	b=np.zeros(numEquations)

	#calculate gradients
	gx=imArr[:,1:imArr.shape[1]]-imArr[:,0:imArr.shape[1]-1]
	gy=imArr[1:imArr.shape[0],:]-imArr[0:imArr.shape[0]-1,:]

	#initialize constants	
	e=0
	s=imArr

	#apply horizontal gradient objective
	for i in range(0,imWidth-1):
		for j in range(0,imHeight):
#			A[e,j*imWidth+i+1]=1
#			A[e,j*imWidth+i]=-1
			A_indices[2*e]=j*imWidth+i+1
			A_indices[2*e+1]=j*imWidth+i
			b[e]=s[j,i+1]-float(s[j,i])
			e+=1

	#apply vertical gradient objective
	for i in range(0,imWidth):
		for j in range(0,imHeight-1):
#			A[e,(j+1)*imWidth+i]=1
#			A[e,j*imWidth+i]=-1
			A_indices[2*e]=(j+1)*imWidth+i
			A_indices[2*e+1]=j*imWidth+i
			b[e]=s[j+1,i]-float(s[j,i])
			e+=1

	#apply constant pixel intensity objective
#	A[e,0]=1
	A_indices[2*e]=0
	b[e]=s[0,0]

	#form the sparse matrix A
	A_indptr=np.hstack((np.arange(0,numIndices,2),numIndices))
	A_data=np.zeros(numIndices)
	for i in range(0,numIndices-1):
		if i%2==0:
			A_data[i]=1
		else:
			A_data[i]=-1
	A_data[numIndices-1]=1
	A=scipy.sparse.csr_matrix((A_data,A_indices,A_indptr),shape=(numEquations,numVariables))

	#solve the least squares problem
#	x = np.linalg.lstsq(A,b)[0]
	x = scipy.sparse.linalg.lsqr(A,b)[0]
	x = x.reshape(imArr.shape)

	#Calculate the error
	print 'Error: ',np.sum((imArr-x)**2)
	'''