#Based on MATLAB code from Debevec and Malik (SIGGRAPH 1997), 
#"Recovering High Dynamic Range Radiance Maps from Photographs"

import numpy as np
import matplotlib.pyplot as plt

#minimum, maximum RGB color values
Zmin = 0
Zmax = 255
Zmid = (Zmin+Zmax)/2
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
			wij = w[Z[i,j]] #do I need the +1?
			A[k,Z[i,j]] = wij #do I need the +1?
			A[k,n+i] = -1*wij
			b[k,0] = -wij * B[j] #originally B[i,j] but B is a vector???
			k += 1

	#Fix the curve
	A[k,128] = 1
	k += 1

	#Include the smoothness equations

	for i in range(0,n-1):
		A[k,i] = l*w[i+1]
		A[k,i+1]= -2*l*w[i+1]
		A[k,i+2]=l*w[i+1]
		k += 1

	#solve the system of equations
	#substitute for A\b
	#x = np.linalg.solve(A,b) #only works for square matrices
	x = np.linalg.lstsq(A,b)[0]
	g = x[0:n]
	lE = x[n:np.size(x,0)]

	return g, lE
	
j = 5
X = np.zeros((4,j))
B = np.ones(j)
l = 2

w=np.ones((n,1))
#for i in range(0,n):
#	if i<=Zmid:
#		w[i]=i-Zmin
#	else:
#		w[i]=Zmax-i

z=np.mat([
[110,230,255],
[60,175,253],
[50,120,250],
[30,90,200],
[20,55,150],
])
#zmin=np.min(z)
#zmax=np.max(z)
#zmid=(zmin+zmax)/2
B=np.array([2,1.33,0.67,0,-0.67]);
l=5
zt=np.transpose(z)
#plt.plot(np.transpose(z).A1,np.concatenate((B,B,B)),'x')
plt.subplot(121)
plt.plot(zt[0].A1,B,'x',zt[1].A1,B,'+',zt[2].A1,B,'o')
plt.axis([0,300,-6,6])
#plt.show()
	
test_output = rfsolve(z,B,l,w)
g=np.mat(test_output[0]).A1
lE=test_output[1]
y=np.zeros((5,3))
yy=np.zeros((5,3))
for i in range(0,5):
	for j in range(0,3):
		y[i,j]=g[z[i,j]]
		yy[i,j]=lE[i]+B[j]
#print y
#print yy
yt=np.transpose(y)
yyt=np.transpose(y)
plt.subplot(122)
#plt.plot(np.transpose(z).A1,np.mat(np.transpose(y)).A1,'x')
plt.plot(zt[0].A1,yt[0],'x',zt[1].A1,yt[1],'+',zt[2].A1,yt[2],'o')
plt.axis([0,300,-6,6])
#plt.subplot(313)
#plt.plot(np.transpose(z).A1,np.mat(np.transpose(yy)).A1,'x')
#plt.plot(zt[0].A1,yyt[0],'x',zt[1].A1,yyt[1],'+',zt[2].A1,yyt[2],'o')
#plt.axis([0,300,-6,6])
plt.show()

#print g
#print lE
#print "g: "
#print test_output[0]
#print "lE: "
#print test_output[1]


	
