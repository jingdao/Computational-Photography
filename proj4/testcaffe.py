import numpy as np
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = '/home/jd/Downloads/caffe-master/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = caffe_root+'models/bvlc_alexnet/deploy.prototxt'
PRETRAINED = caffe_root+'models/bvlc_alexnet/bvlc_alexnet.caffemodel'
IMAGE_FILE = caffe_root+'examples/images/cat.jpg'

caffe.set_mode_cpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED, \
                       mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1), \
                       channel_swap=(2,1,0), \
                       raw_scale=255, \
                       image_dims=(256, 256))

input_image = caffe.io.load_image(IMAGE_FILE)
prediction = net.predict([input_image])
print 'prediction shape:', prediction[0].shape
print 'predicted class:', prediction[0].argmax()

# load labels
imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
# sort top k predictions from softmax output
top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
print labels[top_k]
