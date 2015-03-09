from scipy import ndimage
import numpy as np
import poissonBlending
import matplotlib.pyplot as plt

if __name__=="__main__":
	src=ndimage.imread('samples/penguin_aligned.jpg')
	tgt=ndimage.imread('samples/im2_small.JPG')
	mask_img=ndimage.imread('samples/penguin_mask.jpg')
	useMixedGradient = False

	#get mask: specifices which pixels in the source image are the source region
	#True if pixel in source image is in source region, False otherwise
	mask=np.zeros(mask_img.shape,dtype=bool)
	for i in range(mask_img.shape[0]):
		for j in range(mask_img.shape[1]):
			if mask_img[i,j]>0:
				mask[i,j]=True
			else:
				mask[i,j]=False

	#Calculate Poisson Blend in 3 color channels
	vRed=poissonBlending.poissonBlend(src[:,:,0],tgt[:,:,0],mask,useMixedGradient)
	vGreen=poissonBlending.poissonBlend(src[:,:,1],tgt[:,:,1],mask,useMixedGradient)
	vBlue=poissonBlending.poissonBlend(src[:,:,2],tgt[:,:,2],mask,useMixedGradient)
	
	#round possible floating point values to ints
	vRed = np.array(vRed,dtype=np.uint8)
	vGreen = np.array(vGreen,dtype=np.uint8)
	vBlue = np.array(vBlue,dtype=np.uint8)
	
	#put blended values of source region back into target region
	tgt[:,:,0] = mask*vRed + (1 - mask)*tgt[:,:,0]
	tgt[:,:,1] = mask*vGreen + (1 - mask)*tgt[:,:,1]
	tgt[:,:,2] = mask*vBlue + (1 - mask)*tgt[:,:,2]
	
	plt.imshow(tgt)
	plt.show()

