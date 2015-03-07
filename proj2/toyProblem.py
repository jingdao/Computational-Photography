from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.sparse.linalg

if __name__=="__main__":
	imArr=ndimage.imread('samples/toy_problem.png')
	imWidth=imArr.shape[1]
	imHeight=imArr.shape[0]
	numVariables=imWidth*imHeight
	numEquations=(imWidth-1)*imHeight+(imHeight-1)*imWidth+1
	numIndices=(numEquations-1)*2+1
#	A=np.zeros((numEquations,numVariables))
	A_indices=np.zeros(numIndices,dtype=np.int64)
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

#	print imArr
#	print A
#	print b
#	print x

	#Calculate the error
	print 'Error: ',np.sum((imArr-x)**2)

	plt.imshow(gx,cmap=cm.Greys_r)
	plt.title('Gradient in x direction')
	plt.figure()
	plt.imshow(gy,cmap=cm.Greys_r)
	plt.title('Gradient in y direction')
	plt.figure()
	plt.imshow(x,cmap=cm.Greys_r)
	plt.title('Recovered image')
	plt.show()
