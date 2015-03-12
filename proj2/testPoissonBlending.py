from scipy import ndimage
import numpy as np
import poissonBlending
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def testPoissonBlend(srcFile,tgtFile,maskFile,useMixedGradient):
	src=ndimage.imread(srcFile)
	tgt=ndimage.imread(tgtFile)
	mask_img=ndimage.imread(maskFile)

	#if mask img has multiple channels
	if len(mask_img.shape)>2:
		mask_img=mask_img[:,:,0]


	#get mask: specifices which pixels in the source image are the source region
	#True if pixel in source image is in source region, False otherwise
	mask=np.zeros((mask_img.shape[0],mask_img.shape[1]),dtype=bool)
	mask_max_value=np.max(mask_img)
	for i in range(mask_img.shape[0]):
		for j in range(mask_img.shape[1]):
			if mask_img[i,j]>=mask_max_value/2:
				mask[i,j]=True

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

	#this is the image with only values from the reconstructed source region
	src2 = np.dstack([vRed,vGreen,vBlue])
	
	plt.figure()
	plt.imshow(mask)
	plt.figure()
	plt.imshow(src2)
	plt.figure()
	plt.imshow(tgt[:,:,0],cmap=cm.Greys_r)
	plt.figure()
	plt.imshow(tgt[:,:,1],cmap=cm.Greys_r)
	plt.figure()
	plt.imshow(tgt[:,:,2],cmap=cm.Greys_r)
	plt.figure()
	plt.imshow(tgt)

if __name__=="__main__":
	useMixedGradient = False
#	testPoissonBlend('samples/penguin_aligned.jpg','samples/im2_small.JPG','samples/penguin_mask.jpg',useMixedGradient)
#	testPoissonBlend('samples/penguin_chick_aligned.jpg','samples/im2_small.JPG','samples/penguin_chick_mask.jpg',useMixedGradient)
#	testPoissonBlend('samples/mona-leber-source.jpg','samples/mona-leber-target.jpg','samples/mona-leber-mask.jpg',useMixedGradient)
#	testPoissonBlend('samples/plane_src.jpg','samples/plane_tgt.jpg','samples/plane_mask.jpg',useMixedGradient)
	testPoissonBlend('samples/mixed_src.jpg','samples/mixed_tgt.jpg','samples/mixed_mask.jpg',True)
	plt.show()
