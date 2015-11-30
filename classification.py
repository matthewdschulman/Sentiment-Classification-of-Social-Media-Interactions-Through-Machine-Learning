'''
    Text Sentiment Classification
    AUTHor Matt Schulman and Bahram Banisadr
'''

import numpy as np
import time
import sklearn.metrics
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from tabulate import tabulate
from sklearn.metrics import precision_recall_fscore_support
from sklearn import svm
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from matplotlib.backends.backend_pdf import PdfPages

print "Welcome to the Text Sentiment Classifier..."

def mapToNumericTargetValues(x):
	if x == 'negative':
		return -1
	elif x == 'positive':
		return 1
	elif x == 'neutral':
		return 0
	else:
		print "ERRor WITH MAPPING TO NUMERIC VALS"
		return -5

def appendTrainingDataToEnsemble(ensemble, name, typeOfData, trainingData, trainingTargets):
	ensemble.append([name, typeOfData, trainingData, trainingTargets, 'classifier_placeholder', 'predicted_training_values_placeholder', 'traning_accuracy_placeholder', 'training_accuracy_placeholder', 'relevance_placeholder', 'testing_predictions_placeholder'])
	return ensemble

def printPrediction(classifierName, predictions):
	mood = ''
	if predictions[0] == 1:
		mood = 'positive'
	elif predictions[0] == -1:
		mood = 'negative'
	elif predictions[0] == 0:
		mood = 'neutral'
	print "{0} predicts {1}".format(classifierName, mood)

def getRelevance(dataType, testType):
	if dataType == testType:
		return 1
	elif testType == 'SMS' or testType == 'FB':
		if dataType == 'Tweet':
			return 0.85
		elif dataType == 'SMS' or dataType == 'FB':
			return 1.0
		elif dataType == 'Product Review' or dataType == 'Movie Review':
			return 0.4
	elif testType == 'Tweet':
		if dataType == 'SMS' or dataType == 'FB':
			return 0.85
		elif dataType == 'Movie Review' or dataType == 'Product Review':
			return 0.6
	elif testType == 'Movie Review' or testType == 'Product Review':
		if dataType == 'SMS' or dataType == 'FB':
			return 0.4
		if dataType == 'Tweet':
			return 0.6
	else:
		print "ERRor WITH GET RELEVANCE! Contact Bahram or Matt for support"
		print "Returning a relevance of 1 so the algorithm doesn't crash"
		return 1.0
	

# set up a 2-D array that contains each ensemble classifier.
# Each classifier contains [name, source_type, training_data, target_training_values, classifier, predicted_training_values, training_accuracy, relevance, testing_predictions]
# Data types must fit one of the following types: SMS, Tweet, FB, Movie Review, Product Review
ensemble = []

num_of_data_sets = 0

# import the sets of training data 

# sms-test-gold-A.tsv training data
with open('data/sms/sms-test-gold-A.tsv','r') as f:
	cur_training = [x.strip().split('\t') for x in f]
cur_np_training = np.array(cur_training)
cur_np_training_data = cur_np_training[:,5]
cur_np_training_target = map(mapToNumericTargetValues, cur_np_training[:,4])
ensemble = appendTrainingDataToEnsemble(ensemble, 'SMS-A Naive Bayes', 'SMS', cur_np_training_data, cur_np_training_target)
ensemble = appendTrainingDataToEnsemble(ensemble, 'SMS-A SVM', 'SMS', cur_np_training_data, cur_np_training_target)
num_of_data_sets += 1

# sms-test-gold-B.tsv training data
with open('data/sms/sms-test-gold-B.tsv','r') as f:
	cur_training = [x.strip().split('\t') for x in f]
cur_np_training = np.array(cur_training)
cur_np_training_data = cur_np_training[:,3]
cur_np_training_target = map(mapToNumericTargetValues, cur_np_training[:,2])
ensemble = appendTrainingDataToEnsemble(ensemble, 'SMS-B Naive Bayes', 'SMS', cur_np_training_data, cur_np_training_target)
ensemble = appendTrainingDataToEnsemble(ensemble, 'SMS-B SVM', 'SMS', cur_np_training_data, cur_np_training_target)
num_of_data_sets += 1

# Create classifiers, fit classifiers, predict on training data, compute accuracy on training data
for i in range(0,num_of_data_sets):
	# Naive Bayes
	cur_classifier_index = i*2
	cur_clf_bayes = Pipeline([('vect', CountVectorizer(lowercase=True, stop_words='english')),
                           ('tfidf', TfidfTransformer(norm='l2', sublinear_tf=True)),
		           ('clf', MultinomialNB()),
		          ]
			 )
	cur_clf_bayes = cur_clf_bayes.fit(ensemble[cur_classifier_index][2], ensemble[cur_classifier_index][3])
	ensemble[cur_classifier_index][4] = cur_clf_bayes
	cur_nb_predictions = cur_clf_bayes.predict(ensemble[cur_classifier_index][2])
	ensemble[cur_classifier_index][5] = cur_nb_predictions
	cur_nb_accuracy = np.mean(cur_nb_predictions == ensemble[cur_classifier_index][3])
	ensemble[cur_classifier_index][6] = cur_nb_accuracy
	print "The training accuracy for {0} = {1}".format(ensemble[cur_classifier_index][0], cur_nb_accuracy)

	# SVM
	cur_classifier_index = i*2 + 1
	cur_clf_svm = Pipeline([('vect', CountVectorizer(lowercase=True, stop_words='english')),
                           ('tfidf', TfidfTransformer(norm='l2', sublinear_tf=True)),
 			   ('clf', svm.SVC(kernel=sklearn.metrics.pairwise.linear_kernel, probability=True)), 
		          ]
			 )
	cur_clf_svm = cur_clf_svm.fit(ensemble[cur_classifier_index][2], ensemble[cur_classifier_index][3])
	ensemble[cur_classifier_index][4] = cur_clf_svm
	cur_svm_predictions = cur_clf_svm.predict(ensemble[cur_classifier_index][2])
	ensemble[cur_classifier_index][5] = cur_svm_predictions
	cur_svm_accuracy = np.mean(cur_svm_predictions == ensemble[cur_classifier_index][3])
	ensemble[cur_classifier_index][6] = cur_svm_accuracy
	print "The training accuracy for {0} = {1}".format(ensemble[cur_classifier_index][0], cur_svm_accuracy)


# predict for testing data
while 1:
	print "Please enter a sentence to be classified:"
	user_input = raw_input()
	np_testing_data = [user_input]
	print "Please enter the type of text this is. Please enter 'SMS', 'FB', 'Tweet', 'Movie Review', or 'Product Review'"
	test_type_of_text = raw_input()
	predictions = []
	weights = []
	for i in range(0,num_of_data_sets):
		relevance_for_this_data_set = getRelevance(ensemble[i*2][1], test_type_of_text)

		# Naive Bayes Current Prediction
		cur_classifier_index = i*2
		cur_nb_predictions = ensemble[cur_classifier_index][4].predict(np_testing_data)
		ensemble[cur_classifier_index][8] = cur_nb_predictions
		printPrediction(ensemble[cur_classifier_index][0], cur_nb_predictions)  
		predictions.append(cur_nb_predictions[0])
		weights.append(ensemble[cur_classifier_index][6] * relevance_for_this_data_set) 

		# SVM Current Prediction
		cur_classifier_index = i*2 + 1
		cur_svm_predictions = ensemble[cur_classifier_index][4].predict(np_testing_data)
		ensemble[cur_classifier_index][8] = cur_svm_predictions
		printPrediction(ensemble[cur_classifier_index][0], cur_svm_predictions)  
		predictions.append(cur_svm_predictions[0])
		weights.append(ensemble[cur_classifier_index][6] * relevance_for_this_data_set) 

	weighted_average = np.average(predictions, weights=weights)
	print "predictions = {0}".format(predictions)
	print "weights = {0}".format(weights)
	print "the weighted average = {0}".format(weighted_average)
