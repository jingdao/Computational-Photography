from frameDistances import *
from readImages import *
from video import *
from evaluateTransitions import *

numIters = 10**2
p = 10
alpha = 0.99
tolerance = 10**-3

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
sigma = 2*avgNonzeroDistance
#probMatrix = probabilityMatrix(filteredDists, sigma)
probMatrix = probabilityMatrix(futureDiffs, sigma)
distributions = probabilityDistributions(probMatrix)

#get probabilities

#make video 2x longer
makeVideo(pixelArrayRed,pixelArrayGreen,pixelArrayBlue,numFrames,imHeight,imWidth,fps,distributions)
#saveVideo(pixelArrayRed,pixelArrayGreen,pixelArrayBlue,numFrames,2*numFrames,imHeight,imWidth,fps,distributions)
