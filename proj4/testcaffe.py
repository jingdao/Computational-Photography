# Make sure that caffe is on the python path:
import numpy as np
import os
import sys
caffe_root = '/home/jd/Downloads/caffe-master/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
#MODEL_ZOO = 'bvlc_reference_caffenet'
MODEL_ZOO = 'bvlc_alexnet'
MODEL_FILE = caffe_root+'models/'+MODEL_ZOO+'/deploy.prototxt'
PRETRAINED = caffe_root+'models/'+MODEL_ZOO+'/'+MODEL_ZOO+'.caffemodel'
IMAGE_DIR = caffe_root+'examples/images/'

caffe.set_mode_cpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED, \
                       mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1), \
                       channel_swap=(2,1,0), \
                       raw_scale=255, \
                       image_dims=(256, 256))
## load labels
#imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
#labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

#image_list = []
#image_names = []
features=[]
for f in os.listdir(IMAGE_DIR):
	if os.path.isfile(IMAGE_DIR+f):
		input_image = caffe.io.load_image(IMAGE_DIR+f)
#		image_names.append(f)
#		image_list.append(input_image)
		prediction = net.predict([input_image])
		features.append(prediction[0])

features = np.array(features)
np.save('features.npy',features.transpose())


#prediction = net.predict(image_list)
#for i in range(0,len(image_names)):
#	print 'image:',image_names[i]
#	print 'prediction shape:', prediction[i].shape
#	print 'predicted class:', prediction[i].argmax()
#	# sort top k predictions from softmax output
#	top_k = net.blobs['prob'].data[i].flatten().argsort()[-1:-6:-1]
#	print labels[top_k]
