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
FAKE_TRAIN_DIR = 'clip/train/'
FAKE_TEST_DIR = 'clip/test/'
dirs = [REAL_TRAIN_DIR, REAL_TEST_DIR, FAKE_TRAIN_DIR, FAKE_TEST_DIR]
numImages = [800, 200, 800, 200]

caffe.set_mode_cpu()
net = caffe.Net(MODEL_FILE,PRETRAINED,caffe.TEST)
net.blobs['data'].reshape(1,3,227,227)

features_train=[]
features_test=[]
for j in range(4):
	for i in range(1,numImages[j]+1):
		input_image = np.load(dirs[j] + str(i) + '.npy')
		net.blobs['data'].data[...] = input_image
		prediction = net.forward()
		if j%2==0:
			features_train.append(np.hstack((prediction['prob'][0],1 if j<2 else 0)))
		else:
			features_test.append(np.hstack((prediction['prob'][0],1 if j<2 else 0)))

features_train = np.array(features_train)
features_test = np.array(features_test)
np.random.shuffle(features_train)
np.random.shuffle(features_test)
labels_train = features_train[:,-1]
labels_test = features_test[:,-1]
features_train = features_train[:,:-1]
features_test = features_test[:,:-1]
np.save('features_train.npy',features_train.transpose())
np.save('features_test.npy',features_test.transpose())
np.save('labels_train.npy',labels_train)
np.save('labels_test.npy',labels_test)

