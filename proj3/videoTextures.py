from frameDistances import *
from readImages import *
from video import *
from evaluateTransitions import *

numIters = 10**6
p = 2
alpha = 0.999
tolerance = 10**-2
showIndex = True
crossFade = True

if len(sys.argv)==2:
	dataset=sys.argv[1]
else:
	dataset='clock'
pixelArrayRed,pixelArrayGreen,pixelArrayBlue,numFrames,imHeight,imWidth,fps = getPixelArrayFromFiles(dataset)
#animate(pixelArray,numFrames,imHeight,imWidth)

diffs = diffMatrix(np.array(pixelArrayRed,dtype=np.float))
filteredDists = filterDists(diffs)
futureDiffs = anticipateFutureCosts(filteredDists,numIters,p,alpha,tolerance)

#for use in converting to probabilities
#avgNonzeroDistance = np.sum(filteredDists)/np.count_nonzero(filteredDists)
avgNonzeroDistance = np.sum(futureDiffs)/np.count_nonzero(futureDiffs)
sigma = 0.1*avgNonzeroDistance
#probMatrix = probabilityMatrix(filteredDists, sigma)
probMatrix = probabilityMatrix(futureDiffs, sigma)
distributions = probabilityDistributions(probMatrix)

#get probabilities

#make video 2x longer
#makeVideo(pixelArrayRed,pixelArrayGreen,pixelArrayBlue,numFrames,imHeight,imWidth,fps,distributions)
saveVideo(pixelArrayRed,pixelArrayGreen,pixelArrayBlue,numFrames,2*numFrames,imHeight,imWidth,fps,distributions,showIndex,crossFade)
