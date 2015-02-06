from scipy import misc
import math

f=open('memorial/memorial.hdr_image_list.txt')
f.readline()
f.readline()
f.readline()
numImages=0
numPixels=0
B=[]
for s in f:
	sArray=s.split(' ')
	imFile=sArray[0].replace('.ppm','.png')
	exposureTime=math.log(1/float(sArray[1]))
	B.append(exposureTime)
	imArr=misc.imread('memorial/'+imFile)
	numImages+=1
	numPixels+=imArr.shape[0]*imArr.shape[1]
print numImages,numPixels
print B
f.close()
