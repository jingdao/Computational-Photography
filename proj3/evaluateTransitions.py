import numpy as np

#anticipating the future
#also have the algorithm return if it hasn't converged after a large number of iterations
def anticipateFutureCosts(diffsMatrix,maxIters,p,alpha,tolerance):
	initialMatrix = np.power(diffsMatrix,p)
	diffsMatrix = initialMatrix
	numRows = np.shape(initialMatrix)[0]
	numCols = np.shape(initialMatrix)[1]
	#value of old matrix after last iteration (to test for convergence)
	oldMatrix = np.zeros((numRows,numCols)) 
	
	#keep track of number of iterationsnp.max(np.absolute(np.subtract(diffsMatrix,oldMatrix)))
	numIterations = 0
	
	maxDiff = 1 #definitely above tolerance
	
	#iteratively update algorithm
	while numIterations < maxIters and maxDiff <= tolerance:
		#get indices of minimum elements of each row of array
		minimumRowElementsIndices = diffsMatrix.argmin(axis=1)
		#get minimum elements of each row of array
		minimumRowElements = diffsMatrix[minimumRowElementsIndices]
		
		#update values of diffsMatrix
		diffsMatrix = initialMatrix + alpha*np.hstack(minimumRowElements,numCols)
		
		#see if matrix values have converged
		maxDiff = np.max(np.absolute(np.subtract(diffsMatrix,oldMatrix)))
		if maxDiff <= tolerance:
			break
		
		#otherwise repeat the process
		oldMatrix = diffsMatrix
		numIterations += 1
		
		if numIterations >= maxIters:
			print "Returning before convergence"
			print "Maximum difference between matrix entries after last iteration: ", maxDiff
			
	return diffsMatrix
		