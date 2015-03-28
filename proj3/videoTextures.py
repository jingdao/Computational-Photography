from frameDistances import *
from readImages import *
from video import *

pixelArray,numFrames,imHeight,imWidth = getPixelArrayFromFiles('clock')
#animate(pixelArray,numFrames,imHeight,imWidth)

diffs = diffMatrix(pixelArray)
filteredDists = filterDists(diffs)

#for use in converting to probabilities
avgNonzeroDistance = np.sum(filteredDists)/np.count_nonzero(filteredDists)
sigma = 2*avgNonzeroDistance
probMatrix = probabilityMatrix(filteredDists, sigma)
distributions = probabilityDistributions(probMatrix)

#get probabilities

#make video 2x longer
makeVideo(pixelArray,2*numFrames,imHeight,imWidth, distributions)