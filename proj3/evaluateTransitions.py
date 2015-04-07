import numpy as np
import matplotlib.pyplot as plt

#anticipating the future
#also have the algorithm return if it hasn't converged after a large number of iterations
def anticipateFutureCosts(diffsMatrix,maxIters,p,alpha,tolerance):
	initialMatrix = np.power(diffsMatrix,p)
	diffsMatrix = initialMatrix
	numRows = np.shape(initialMatrix)[0]
	numCols = np.shape(initialMatrix)[1]
	#value of old matrix after last iteration (to test for convergence)
	oldMatrix = np.zeros((numRows,numCols)) 
	origDiffsMatrix = diffsMatrix
	
	#keep track of number of iterationsnp.max(np.absolute(np.subtract(diffsMatrix,oldMatrix)))
	numIterations = 0
	
	maxDiff = 1 #definitely above tolerance
	
	#iteratively update algorithm
	while numIterations < maxIters and maxDiff >= tolerance:
		#form matrix whose columns are minimum row elements
		minimumRowElements = []
		largeElt = np.max(diffsMatrix) #definitely bigger than minimum element
		for row in range(0,numRows): #find minimum element not on the diagonal
			minimumElt = largeElt
			for col in range(0,numCols):
				if col != row and diffsMatrix[row,col] < minimumElt:
					minimumElt = diffsMatrix[row,col]
			minimumRowElements.append(minimumElt)
		
		minimumRowEltsVector = np.array(minimumRowElements)
		minimumRowEltsMatrix = np.matrix(minimumRowEltsVector)
		for i in range(0,numCols-1):
			minimumRowEltsMatrix = np.vstack((minimumRowEltsMatrix,minimumRowEltsVector))
		
		#update values of diffsMatrix
		diffsMatrix = initialMatrix + alpha*minimumRowEltsMatrix
		
		#see if matrix values have converged
		maxDiff = np.max(np.absolute(np.subtract(diffsMatrix,oldMatrix)))
		if maxDiff <= tolerance:
			break
		
		#otherwise repeat the process
		oldMatrix = diffsMatrix
		numIterations += 1

	plt.imshow(diffsMatrix)
	plt.show()	
	return diffsMatrix
	
if __name__ == "__main__":
	anticipateFutureCosts( np.matrix([[0,1],[1,0]]), 10**3, 2, 0.995, 10**-3)
		