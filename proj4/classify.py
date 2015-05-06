import numpy as np
from sklearn import linear_model,svm

#trains a machine learning classifier on a training data set
#supervised: also takes in training labels
#returns a classifier (a hypothesis)
def train_classifier(training_data, training_labels):
   #kernelized SVM for classification
   #classifier = svm.SVC(kernel='rbf') #leave blank or 'linear' for linear SVM
   classifier = linear_model.LogisticRegression()
   classifier.fit(training_data, training_labels)
   return classifier

#compute predictions for figuring out training error or making test predictions
#takes in a machine learning hypothesis and makes predictions on it
def make_predictions(classifier, data):
   predictions = classifier.predict(data)
   return predictions

def predict_probs(classifier, data):
   predict_probs = classifier.predict_proba(data)
   return predict_probs
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

   #load 8000 images, 4000 real and 4000 fake, each with 1000 features
   #this is subdivided into 6400 for training data and 1600 for test data
   features_train8 = np.load('features/features_train_fc8.npy')
   features_test8 = np.load('features/features_test_fc8.npy') 
   labels_train =  np.load('features/labels_train_fc8.npy') 
   labels_test = np.load('features/labels_test_fc8.npy') 
   imagenames_train = np.load('features/imagenames_train_fc8.npy')
   imagenames_test = np.load('features/imagenames_test_fc8.npy') 
   
   #'''
   features_train7 = np.load('features/features_train_fc7.npy')
   features_test7 = np.load('features/features_test_fc7.npy')

   #features_train6 = np.load('features/features_train_fc6.npy')
   #features_test6 = np.load('features/features_test_fc6.npy')

   features_train = np.concatenate((features_train8, features_train7),axis=1) 
   features_test = np.concatenate((features_test8, features_test7),axis=1)
   #'''
   
   numTrain = features_train.shape[0]
   numTest = features_test.shape[0]
   numFeatures = features_test.shape[1]
   print("There are %d train and %d test points with %d features" % (numTrain, numTest, numFeatures)) 

   trained_classifier = train_classifier(features_train, labels_train)
   train_predictions = make_predictions(trained_classifier, features_train)
   train_error = compute_error(train_predictions, labels_train)
   print("Training error: %d" % train_error)

   test_predictions = make_predictions(trained_classifier, features_test)
   test_error = compute_error(test_predictions, labels_test)
   print("Test error: %f" % test_error)
