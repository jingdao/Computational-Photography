from frameDistances import *
from readImages import *
from video import *

pixelArrayRed,pixelArrayGreen,pixelArrayBlue,numFrames,imHeight,imWidth = getPixelArrayFromFiles('clock')
#animate(pixelArray,numFrames,imHeight,imWidth)

diffs = diffMatrix(pixelArrayRed)
filteredDists = filterDists(diffs)

#for use in converting to probabilities
avgNonzeroDistance = np.sum(filteredDists)/np.count_nonzero(filteredDists)
sigma = 2*avgNonzeroDistance
probMatrix = probabilityMatrix(filteredDists, sigma)
distributions = probabilityDistributions(probMatrix)

#get probabilities

#make video 2x longer
makeVideo(pixelArrayRed,pixelArrayGreen,pixelArrayBlue,numFrames,imHeight,imWidth, distributions)
