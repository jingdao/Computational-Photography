from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.sparse.linalg

#filters out the lowest n elements from an array where r=n/size
def lowFilter(X,r):
	S=set()
	for i in np.nditer(X):
		S.add(float(i))
	S_sorted = sorted(S)
	n = int(r*len(S_sorted))
	threshold = S_sorted[n]
	for i in np.nditer(X,op_flags=['readwrite']):
		if i<threshold:
			i[...]=0



#reconstructs an image from its gradients
def solvePoisson(arr,width,height,A):
	b=np.zeros(numEquations)
	e=0

	#calculate gradients
	gx=np.array(arr[:,1:arr.shape[1]],dtype=float)-arr[:,0:arr.shape[1]-1]
	gy=np.array(arr[1:arr.shape[0],:],dtype=float)-arr[0:arr.shape[0]-1,:]

	plt.figure()
	plt.imshow(gx,cmap=cm.Greys_r)
	#filter out gradients on non-edges (low values)
	lowFilter(gx,0)
	lowFilter(gy,0)

	plt.figure()
	plt.imshow(gx,cmap=cm.Greys_r)


	#apply horizontal gradient objective
	for i in range(0,width-1):
		for j in range(0,height):
			b[e]=gx[j,i]
			e+=1

	#apply vertical gradient objective
	for i in range(0,width):
		for j in range(0,height-1):
			b[e]=gy[j,i]
			e+=1

	#apply constant pixel intensity objective
	b[e]=arr[0,0]

	#solve the least squares problem
	x = scipy.sparse.linalg.lsqr(A,b)[0]
	x = x.reshape((height,width))

	return x
	

if __name__=="__main__":
	imArr=ndimage.imread('samples/boy.png')
	imWidth=imArr.shape[1]
	imHeight=imArr.shape[0]
	numVariables=imWidth*imHeight
	numEquations=(imWidth-1)*imHeight+(imHeight-1)*imWidth+1
	numIndices=(numEquations-1)*2+1
	A_indices=np.zeros(numIndices,dtype=np.int64)

	#initialize constants	
	e=0

	#apply horizontal gradient objective
	for i in range(0,imWidth-1):
		for j in range(0,imHeight):
			A_indices[2*e]=j*imWidth+i+1
			A_indices[2*e+1]=j*imWidth+i
			e+=1

	#apply vertical gradient objective
	for i in range(0,imWidth):
		for j in range(0,imHeight-1):
			A_indices[2*e]=(j+1)*imWidth+i
			A_indices[2*e+1]=j*imWidth+i
			e+=1

	#apply constant pixel intensity objective
	A_indices[2*e]=0

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

	print "start solving"
	vRed=solvePoisson(imArr[:,:,0],imWidth,imHeight,A)
	print "red"
	vGreen=solvePoisson(imArr[:,:,1],imWidth,imHeight,A)
	print "green"
	vBlue=solvePoisson(imArr[:,:,2],imWidth,imHeight,A)
	print "blue"

	finalImage=np.dstack([vRed,vGreen,vBlue])
	finalImage=np.array(finalImage,dtype=np.uint8)

	plt.figure()
	plt.imshow(imArr)
	plt.title('Original image')
	plt.figure()
	plt.imshow(finalImage)
	plt.title('Recovered image')
	plt.show()
