from scipy import ndimage
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys

#	get pixel values from an image
#	fileName: file name of image
#	pixelArray: 2D array of pixel values
def getPixelArrayFromFiles(fileName):
	imArr=ndimage.imread(fileName)
	pixelArrayRed = imArr[:,:,0]
	pixelArrayGreen = imArr[:,:,1]
	pixelArrayBlue = imArr[:,:,2]
	return pixelArrayRed,pixelArrayGreen,pixelArrayBlue

#	returns the magnitude of the two-dimensional Fast Fourier Transform of A
#	A: a 2D input array
def calculateFFT(A):
	return np.abs(np.fft.fft2(A))

#using the model Power = A/(f^(2+eta))
#				ln(Power) = ln(A) - (2+eta) * ln(f)
#returns eta which tends to zero for natural images
def powerLawFit(P,fmax_x,fmax_y):
	x=np.zeros(fmax_y*fmax_x)
	y=np.zeros(fmax_y*fmax_x)
	k=0
	for i in range(0,fmax_y):
		for j in range(0,fmax_x):
			if not (i==0 and j==0):
				x[k]=np.log(np.sqrt(i*i+j*j))
				y[k]=P[i,j]
				k+=1
	slope, intercept, r_value, p_value, std_err = stats.linregress(x[1:],y[1:])
	print P[0,0], slope, intercept, r_value*r_value
	return r_value*r_value

if __name__=="__main__":
	if len(sys.argv)==2:
		dataset=sys.argv[1]
	else:
		dataset='real/16_24_7_web.jpg'
	pixelArrayRed,pixelArrayGreen,pixelArrayBlue = getPixelArrayFromFiles(dataset)
	imWidth = pixelArrayRed.shape[1]
	imHeight = pixelArrayRed.shape[0]
	transform = calculateFFT(pixelArrayRed)
	logTransform = np.log(transform)
	fmax_x=imWidth/2
	fmax_y=imHeight/2
	rsquared = powerLawFit(logTransform,fmax_x,fmax_y)
	rsquared = powerLawFit(np.log(calculateFFT(pixelArrayGreen)),fmax_x,fmax_y)
	rsquared = powerLawFit(np.log(calculateFFT(pixelArrayBlue)),fmax_x,fmax_y)
#	plt.figure()
#	plt.imshow(logTransform[0:fmax_y,0:fmax_x])
#	plt.colorbar()
#	plt.figure()
#	plt.imshow(pixelArrayRed)
#	plt.colorbar()
#	plt.show()

