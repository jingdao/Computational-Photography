# Make sure that caffe is on the python path:
import numpy as np
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
#transformer = caffe.io.Transformer({'data':(1,3,256,256)})
transformer.set_transpose('data', (2,0,1))
mn = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mn=mn.mean(1).mean(1)
transformer.set_mean('data', mn) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

REAL_TRAIN_DIR = 'real/train/'
REAL_TEST_DIR = 'real/test/'
FAKE_TRAIN_DIR = 'clip/train/'
FAKE_TEST_DIR = 'clip/test/'
dirs = [REAL_TRAIN_DIR, REAL_TEST_DIR, FAKE_TRAIN_DIR, FAKE_TEST_DIR]
numImages = [800, 200, 800, 200]
for j in range(4):
	i = 0
	d = dirs[j].split('/')[0] + '/'
	for f in os.listdir(d):
		if os.path.isfile(d+f) and (f.endswith('.jpg') or f.endswith('.png')):
			i = i+1
			input_image = caffe.io.load_image(d+f)
			if np.max(input_image) > 1:
				print 'warning '+f
			input_image = transformer.preprocess('data',input_image)
			np.save(dirs[j] + str(i) + '.npy',input_image)
#			net.blobs['data'].data[...] = input_image
#			out = net.forward()
			if i>=numImages[j]:
				break
