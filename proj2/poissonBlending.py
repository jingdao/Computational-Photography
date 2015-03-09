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
	b = list() #list of equations for the vector b
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
						
							b.append(neighbor - originalPixelValue)
							e += 1
						if m + n > 0 and not mask[j+n,i+m]:
							ArowIndices.append(e) #the e-th equation
							AcolIndiceappend(j*sourceWidth + i) #which entry
							Adata.append(1) #v_i
						
							neighbor = source[j+n,i+m]
						
							#target image pixel value is a constant with respect to x, what we're solving for
							b.append(neighbor - originalPixelValue + target[(j+n),(i+m)])
							e += 1
						
	#shape of the matrix is (number of equations) x (number of pixels in source/target image)
	A=scipy.sparse.csr_matrix((Adata,(ArowIndices,AcolIndices)),shape=(e,sourceWidth * sourceHeight))

	#solve the least squares problem
#	x = np.linalg.lstsq(A,b)[0]
	x = scipy.sparse.linalg.lsqr(A,b)[0]
	x = x.reshape(sourceArr.shape)
	
	print("x, the reconstruction: ")
	print x
	print("A, the equations: ")
	print A
	print("b, the solutions: ")
	print b		

						
			
	
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
if __name__ == "main":
	source = np.matrix('1 2; 3 4')
	target = np.matrix('3 1; 3 2')
	mask = np.matrix('1 0; 0 0')
	poissonBlend(source, target, mask)