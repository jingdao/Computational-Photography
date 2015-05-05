# Make sure that caffe is on the python path:
import numpy as np
from classify import *
import os
import sys
caffe_root = '/home/jd/Downloads/caffe-master/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')

import caffe

MODEL_ZOO = 'bvlc_alexnet'
MODEL_FILE = caffe_root+'models/'+MODEL_ZOO+'/deploy.prototxt'
PRETRAINED = caffe_root+'models/'+MODEL_ZOO+'/'+MODEL_ZOO+'.caffemodel'
caffe.set_mode_cpu()
net = caffe.Net(MODEL_FILE,PRETRAINED,caffe.TEST)
net.blobs['data'].reshape(1,3,227,227)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
mn = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mn=mn.mean(1).mean(1)
transformer.set_mean('data', mn) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

features_train = np.load('features_train_fc7_pr.npy')
labels_train =  np.load('labels_train_fc7_pr.npy')
trained_classifier = train_classifier(features_train, labels_train)

def predict(fileName):
	try:
		input_image = caffe.io.load_image(fileName)
		input_image = transformer.preprocess('data',input_image)
		net.blobs['data'].data[...] = input_image
		prediction = net.forward()
		output_features = net.blobs['fc7'].data[0]
		train_predictions = make_predictions(trained_classifier, output_features)
		if train_predictions[0] > 0:
			return "real"
		else:
			return "fake"
	except IOError:
		return "invalid image"

if __name__=="__main__":
	if len(sys.argv)==2:
		print predict(sys.argv[1])
