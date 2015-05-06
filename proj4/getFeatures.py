# Make sure that caffe is on the python path:
import numpy as np
import os
import sys
caffe_root = '/home/jd/Downloads/caffe-master/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_ZOO = 'bvlc_alexnet'
MODEL_FILE = caffe_root+'models/'+MODEL_ZOO+'/deploy.prototxt'
PRETRAINED = caffe_root+'models/'+MODEL_ZOO+'/'+MODEL_ZOO+'.caffemodel'
REAL_TRAIN_DIR = 'real/train/'
REAL_TEST_DIR = 'real/test/'
FAKE_TRAIN_DIR = 'fake/train/'
FAKE_TEST_DIR = 'fake/test/'
CLIP_TRAIN_DIR = 'clip/train/'
CLIP_TEST_DIR = 'clip/test/'
dirs = [REAL_TRAIN_DIR, REAL_TEST_DIR, FAKE_TRAIN_DIR, FAKE_TEST_DIR, CLIP_TRAIN_DIR, CLIP_TEST_DIR]
numImages = [4800, 1200, 1600, 400, 3200,800]

caffe.set_mode_cpu()
net = caffe.Net(MODEL_FILE,PRETRAINED,caffe.TEST)
net.blobs['data'].reshape(1,3,227,227)

features_train=[]
features_test=[]
imagenames_train=[]
imagenames_test=[]
for j in range(len(dirs)):
	f = open(dirs[j]+'imagenames.txt')
	for i in range(1,numImages[j]+1):
		print i
		input_image = np.load(dirs[j] + str(i) + '.npy')
		imageName = f.readline()[:-1]
		net.blobs['data'].data[...] = input_image
		prediction = net.forward()
		output_features = net.blobs['fc8'].data[0]
		if j%2==0:
			imagenames_train.append(imageName)
			features_train.append(np.hstack((output_features,1 if j<2 else 0,i-1)))
		else:
			imagenames_test.append(imageName)
			features_test.append(np.hstack((output_features,1 if j<2 else 0,i-1)))
	f.close()

features_train = np.array(features_train)
features_test = np.array(features_test)
np.random.shuffle(features_train)
np.random.shuffle(features_test)
labels_train = features_train[:,-2]
labels_test = features_test[:,-2]
features_train_save = features_train[:,:-2]
features_test_save = features_test[:,:-2]
np.save('features_train_fc8_combined.npy',features_train_save)
np.save('features_test_fc8_combined.npy',features_test_save)
np.save('labels_train_fc8_combined.npy',labels_train)
np.save('labels_test_fc8_combined.npy',labels_test)

imagenames_train_save = []
imagenames_test_save = []

for arr in features_train:
	if (arr[-2]>0.5):
		imagenames_train_save.append(imagenames_train[int(arr[-1])])
	else:
		imagenames_train_save.append(imagenames_train[int(arr[-1]) + numImages[0]])
	
for arr in features_test:
	if (arr[-2]>0.5):
		imagenames_test_save.append(imagenames_test[int(arr[-1])])
	else:
		imagenames_test_save.append(imagenames_test[int(arr[-1]) + numImages[1]])
	
np.save('imagenames_train_fc8_combined.npy',imagenames_train_save)
np.save('imagenames_test_fc8_combined.npy',imagenames_test_save)

