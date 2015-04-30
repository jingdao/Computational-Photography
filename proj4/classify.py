import numpy as np
from sklearn import svm

#compute features from raw images
#takes in a matrix of raw images
#returns a dxn matrix, where n is # of images (d features per image, all in a column)
def compute_features(images):
   pass

#trains a machine learning classifier on a training data set
#supervised: also takes in training labels
#returns a classifier (a hypothesis)
def train_classifier(training_data, training_labels):
   pass

#test on test data
#takes in a machine learning hypothesis and test data
def test_classifier(classifier, test_data):
   pass

#takes in a set of predictions and true labels and computes error
#could be used to compute training or test error
def compute_error(predictions, labels):
   pass

#take in images
if __name__ == "__main__":
   pass
   #sketch of process:
   #images, labels, given (training and test)
   #tr_data = compute_features(images)
   #clf = train_classifier(tr_data,labels)
   #test_preds = test_classifier(clf,test_images)
   #print "Error: ", compute_error(test_preds,test_labels)
