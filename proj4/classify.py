import numpy as np
from sklearn import svm

#trains a machine learning classifier on a training data set
#supervised: also takes in training labels
#returns a classifier (a hypothesis)
def train_classifier(training_data, training_labels):
   #support vector classification with linear kernel
   classifier = svm.SVC()
   classifier.fit(training_data, training_labels)
   return classifier

#test on test data
#takes in a machine learning hypothesis and test data
def test_classifier(classifier, test_data):
   predictions = classifier.predict(test_data)
   return predictions

#takes in a set of predictions and true labels and computes error
#could be used to compute training or test error
def compute_error(predictions, labels):
   differences = np.subtract(predictions,labels)
   num_errors = np.count_nonzero(differences)
   num_obs = predictions.size
   error_rate = float(num_errors)/num_obs
   return error_rate

#take in images
if __name__ == "__main__":

   #load 2000 images, 1000 real and 1000 fake, each with 1000 features
   #this is subdivided into 1600 for training data and 400 for test data
   features_train = np.load('features_train.npy') #(1000,1600)
   features_train = features_train.transpose()
   features_test = np.load('features_test.npy') #(1000,400)
   features_test = features_test.transpose()
   labels_train =  np.load('labels_train.npy') #(1600)
   labels_test = np.load('labels_test.npy') #(400,)i
   
   trained_classifier = train_classifier(features_train, labels_train)
   test_predictions = test_classifier(trained_classifier, features_test)
   test_error = compute_error(test_predictions, labels_test)
   print("Error rate: %f" % test_error)
