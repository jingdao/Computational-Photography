from django.conf.urls import patterns, include, url
from django.conf import settings
from django.conf.urls.static import static
from django.http import HttpResponse
from django.template import RequestContext, loader, Context, Template
# Make sure that caffe is on the python path:
import numpy as np
from sklearn import linear_model,svm
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

#trains a machine learning classifier on a training data set
#supervised: also takes in training labels
#returns a classifier (a hypothesis)
def train_classifier(training_data, training_labels):
	#kernelized SVM for classification
	classifier = linear_model.LogisticRegression()
	classifier.fit(training_data, training_labels)
	return classifier

#compute predictions for figuring out training error or making test predictions
#takes in a machine learning hypothesis and makes predictions on it
def make_predictions(classifier, test_data):
	predictions = classifier.predict(test_data)
	return predictions

def predict_probs(classifier, data):
	predict_probs = classifier.predict_proba(data)
	return predict_probs

features_train = np.load('/home/jd/Documents/555/proj4/features_train_fc7_combined.npy')
labels_train =  np.load('/home/jd/Documents/555/proj4/labels_train_fc7_combined.npy')
trained_classifier = train_classifier(features_train, labels_train)

def predict(fileName):
	try:
		input_image = caffe.io.load_image(fileName)
		input_image = transformer.preprocess('data',input_image)
		net.blobs['data'].data[...] = input_image
		prediction = net.forward()
		output_features = net.blobs['fc7'].data[0]
#		train_predictions = make_predictions(trained_classifier, output_features)
#		if train_predictions[0] > 0:
#			return "REAL"
#		else:
#			return "FAKE"
		prob = predict_probs(trained_classifier, output_features)
		return prob
	except IOError:
		return -1

def onFileUpload(req):
	for key, file in req.FILES.items():
		dest = open('img', 'w')
		dest.write(file.read())
		dest.close()
	template = loader.get_template('index.html')
	p = predict('img')
	prob = p[0][1]
	msg = "ERROR" if prob < 0 else "REAL" if prob > 0.5 else "FAKE"
	res = template.render(RequestContext(req,{"result": msg, "prob": "%6.4f" % (prob*100)}))
	return HttpResponse(res)

def onGetImage(req):
	response = HttpResponse(content_type="image/jpeg")
	f = open('img','r')
	response.write(f.read())
	return response

def foo(req):
	template = loader.get_template('index.html')
	return HttpResponse(template.render(RequestContext(req)))

urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'web.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),
	url(r'^$',foo),
	url(r'^upload',onFileUpload),
	url(r'^img',onGetImage),
) + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
