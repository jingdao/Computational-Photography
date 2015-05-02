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

caffe.set_mode_cpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED, \
                       mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1), \
                       channel_swap=(2,1,0), \
                       raw_scale=255, \
                       image_dims=(256, 256))

features_train=[]
features_test=[]
for f in os.listdir(REAL_TRAIN_DIR):
	if os.path.isfile(REAL_TRAIN_DIR+f):
		input_image = caffe.io.load_image(REAL_TRAIN_DIR+f)
		prediction = net.predict([input_image])
		features_train.append(np.hstack((prediction[0],1)))
for f in os.listdir(REAL_TEST_DIR):
	if os.path.isfile(REAL_TEST_DIR+f):
		input_image = caffe.io.load_image(REAL_TEST_DIR+f)
		prediction = net.predict([input_image])
		features_test.append(np.hstack((prediction[0],1)))
for f in os.listdir(FAKE_TRAIN_DIR):
	if os.path.isfile(FAKE_TRAIN_DIR+f):
		input_image = caffe.io.load_image(FAKE_TRAIN_DIR+f)
		prediction = net.predict([input_image])
		features_train.append(np.hstack((prediction[0],0)))
for f in os.listdir(FAKE_TEST_DIR):
	if os.path.isfile(FAKE_TEST_DIR+f):
		input_image = caffe.io.load_image(FAKE_TEST_DIR+f)
		prediction = net.predict([input_image])
		features_test.append(np.hstack((prediction[0],0)))

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

