#Based on MATLAB code from Debevec and Malik (SIGGRAPH 1997), 
#"Recovering High Dynamic Range Radiance Maps from Photographs"

import numpy as np

#minimum, maximum RGB color values
Zmin = 0
Zmax = 255

n = 256

#arguments:
#Z(i,j): pixel values of of pixel location number i in image j
#B(j): log delta t, or log shutter speed, for image j
#l: lambda, the constant that determines the amount of smoothness
#w(z): weighting function value for pixel value z

#returns:
#g(z): log exposure corresponding to pixel value z
#lE(i) is the log film irradiance at pixel location i

def rfsolve(Z,B,l,w):
	#print "term 1: ", np.size(Z,1)#*np.size(Z,2)+n+1
	A = np.zeros( (np.size(Z,0)*np.size(Z,1)+n+1,n+np.size(Z,0)) );
	b = np.zeros( (np.size(A,0),1) ) #or l?
	
	#Include the data fitting equations
	k = 0
	for i in range(0,np.size(Z,0)):
		for j in range(0,np.size(Z,1)):
			wij = w(Z[i,j]+1) #do I need the +1?
			A[k,Z[i,j]+1] = wij #do I need the +1?
			A[k,n+i] = -1*wij
			b[k,0] = wij * B[j] #originally B[i,j] but B is a vector???
			k += 1

	#Fix the curve
	A[k,129] = 1
	k += 1

	#Include the smoothness equations

	for i in range(0,n-1):
		A[k,i] = l*w(i+1)
		A[k,i+1]= -2*l*w(i+1)
		A[k,i+2]=l*w(i+1);
		k += 1

	#solve the system of equations
	#substitute for A\b
	#x = np.linalg.solve(A,b) #only works for square matrices
	x = np.linalg.lstsq(A,b)

	g = x[0:n]
	lE = x[n+1:np.size(x,0)]

	return g, lE
	
j = 5
X = np.zeros((4,j))
B = np.ones(j)
l = 2

def w(z):
	return 1
	
test_output = rfsolve(X,B,l,w)
print "g: "
print test_output[0]
print "lE: "
print test_output[1]


	