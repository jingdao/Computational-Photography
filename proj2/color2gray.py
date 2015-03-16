from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
import poissonBlending

if __name__=="__main__":

	imRGB=ndimage.imread("samples/colorBlindTest35.png")
	
	if imRGB.shape[0]>imRGB.shape[1]:
		imRGB=imRGB[0:imRGB.shape[1],:,:]
	else:
		imRGB=imRGB[:,0:imRGB.shape[0],:]

	imRGB_float=np.dstack((1.0*imRGB[:,:,0]/255,1.0*imRGB[:,:,1]/255,1.0*imRGB[:,:,2]/255))
	imHSV=matplotlib.colors.rgb_to_hsv(imRGB_float)
	imGrayScale=np.mean(imRGB,axis=2)

	gradient_hue=(imHSV[:imHSV.shape[0]-1,0:imHSV.shape[1]-1,0]-imHSV[:imHSV.shape[0]-1,1:imHSV.shape[1],0])**2 + \
				(imHSV[0:imHSV.shape[0]-1,:imHSV.shape[0]-1,0]-imHSV[1:imHSV.shape[0],:imHSV.shape[0]-1,0])**2
	gradient_saturation=(imHSV[:imHSV.shape[0]-1,0:imHSV.shape[1]-1,1]-imHSV[:imHSV.shape[0]-1,1:imHSV.shape[1],1])**2 + \
				(imHSV[0:imHSV.shape[0]-1,:imHSV.shape[0]-1,1]-imHSV[1:imHSV.shape[0],:imHSV.shape[0]-1,1])**2
	gradient_value=(imHSV[:imHSV.shape[0]-1,0:imHSV.shape[1]-1,2]-imHSV[:imHSV.shape[0]-1,1:imHSV.shape[1],2])**2 + \
				(imHSV[0:imHSV.shape[0]-1,:imHSV.shape[0]-1,2]-imHSV[1:imHSV.shape[0],:imHSV.shape[0]-1,2])**2

	mask=np.ones((imRGB.shape[0]-1,imRGB.shape[1]-1))
	mask=np.vstack((np.hstack((mask,np.zeros((imRGB.shape[0]-1,1)))),np.zeros((1,imRGB.shape[1]))))

	#use max gradient between saturation and value channels 
	finalImage = poissonBlending.poissonBlend(imHSV[:,:,1],imHSV[:,:,2],mask,True)
	finalImage=finalImage*mask+imHSV[:,:,2]*(1-mask)

#	plt.figure()
#	plt.imshow(imRGB[:,:,0],cmap=cm.Greys_r)
#	plt.figure()
#	plt.imshow(imRGB[:,:,1],cmap=cm.Greys_r)
#	plt.figure()
#	plt.imshow(imRGB[:,:,2],cmap=cm.Greys_r)
#	plt.figure()
#	plt.imshow(imHSV[:,:,0],cmap=cm.Greys_r)
#	plt.figure()
#	plt.imshow(imHSV[:,:,1],cmap=cm.Greys_r)
#	plt.figure()
#	plt.imshow(imHSV[:,:,2],cmap=cm.Greys_r)
#	plt.figure()
#	plt.imshow(gradient_hue,cmap=cm.Greys_r)
#	plt.figure()
#	plt.imshow(gradient_saturation,cmap=cm.Greys_r)
#	plt.figure()
#	plt.imshow(gradient_value,cmap=cm.Greys_r)
#	plt.figure()
#	plt.imshow(imGrayScale,cmap=cm.Greys_r)
	plt.figure()
	plt.imshow(finalImage,cmap=cm.Greys_r)

	plt.show()

