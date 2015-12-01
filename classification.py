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
from sklearn.metrics import precision_recall_fscore_support
from sklearn import svm
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from matplotlib.backends.backend_pdf import PdfPages

print "Welcome to the Text Mental Health Classifier..."

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

def appendTrainingDataToEnsemble(ensemble, name, typeOfData, trainingData, trainingTargets, validationData, validationTargets):
	ensemble.append([name, typeOfData, trainingData, trainingTargets, 'classifier_placeholder', 'predicted_training_values_placeholder', 'traning_accuracy_placeholder', 'relevance_placeholder', 'testing_predictions_placeholder', validationData, validationTargets])
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
	
def mapToNumericTypeOfText(x):
	if x == 'SMS':
		return 0
	elif x == 'FB':
		return 1
	elif x == 'Tweet':
		return 2
	elif x == 'Movie Review':
		return 3
	elif x == 'Product Review':
		return 4
	else:
		print "ERROR WITH MAPPING TYPE OF TEXT TO NUMERIC VALS"
		return -5

# set up a 2-D array that contains each ensemble classifier.
# Each classifier contains [name, source_type, training_data, target_training_values, classifier, predicted_training_values, training_accuracy, relevance, testing_predictions, validation_data, validation_targets]
# Data types must fit one of the following types: SMS, Tweet, FB, Movie Review, Product Review
ensemble = []

num_of_data_sets = 0

# import the sets of training data 

# sms-test-gold-A.tsv training data
with open('data/sms/sms-test-gold-A.tsv','r') as f:
	cur_training = [x.strip().split('\t') for x in f]
cur_np_training = np.array(cur_training)
split = cur_np_training.shape[0]*4/5
print "split for A = {0}".format(split)
cur_np_training_data = cur_np_training[:split,5]
cur_np_validation_data = cur_np_training[(split+1):,5]
print cur_np_training_data.shape
print cur_np_validation_data.shape
cur_np_training_target = map(mapToNumericTargetValues, cur_np_training[:split,4])
cur_np_validation_target = map(mapToNumericTargetValues, cur_np_training[(split+1):,4])
ensemble = appendTrainingDataToEnsemble(ensemble, 'SMS-A Naive Bayes', 'SMS', cur_np_training_data, cur_np_training_target, cur_np_validation_data, cur_np_validation_target)
ensemble = appendTrainingDataToEnsemble(ensemble, 'SMS-A SVM', 'SMS', cur_np_training_data, cur_np_training_target, cur_np_validation_data, cur_np_validation_target)
num_of_data_sets += 1

# sms-test-gold-B.tsv training data
with open('data/sms/sms-test-gold-B.tsv','r') as f:
	cur_training = [x.strip().split('\t') for x in f]
cur_np_training = np.array(cur_training)
split = cur_np_training.shape[0]*4/5
print "split for B = {0}".format(split)
cur_np_training_data = cur_np_training[:split,3]
cur_np_validation_data = cur_np_training[(split+1):,3]
cur_np_training_target = map(mapToNumericTargetValues, cur_np_training[:split,2])
cur_np_validation_target = map(mapToNumericTargetValues, cur_np_training[(split+1):,2])
ensemble = appendTrainingDataToEnsemble(ensemble, 'SMS-B Naive Bayes', 'SMS', cur_np_training_data, cur_np_training_target, cur_np_validation_data, cur_np_validation_target)
ensemble = appendTrainingDataToEnsemble(ensemble, 'SMS-B SVM', 'SMS', cur_np_training_data, cur_np_training_target, cur_np_validation_data, cur_np_validation_target)
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


# Create feature matrix for training of second layer
# second layer is an nx(1+8*num_of_data_sets+1) matrix
# n is the total number of training points
# the columns in second_layer are features as follows: [numeric mapping of type of text, NB weight, NB neg proba, NB neutral proba, NB pos proba, SVM weight, SVM neg proba, SVM neutral proba, SVM pos proba . . ., target value]

second_layer = []
print "num_of_data_sets = {0}".format(num_of_data_sets)

for set_num in range(0,num_of_data_sets):
	print "set_num = {0}".format(set_num)

	test_type_of_text = ensemble[set_num*2][1]
	#validation_data = ensemble[set_num*2][9]
	#validation_targets = ensemble[set_num*2][10]
	validation_data = ensemble[set_num*2][2]
	validation_targets = ensemble[set_num*2][3]
	num_data_points = validation_data.shape[0]

	for data_row in range(0,num_data_points):

		np_testing_data = validation_data[data_row]

		second_layer_row = []
		second_layer_row.append(mapToNumericTypeOfText(test_type_of_text))
		for i in range(0,num_of_data_sets):
			relevance_for_this_data_set = getRelevance(ensemble[i*2][1], test_type_of_text)

			# Naive Bayes Current Prediction
			cur_classifier_index = i*2
			print "name = {0} | cur_classifier_index = {1} | data_row = {2} | np_testing_data = {3}".format(ensemble[cur_classifier_index][0], cur_classifier_index, data_row, np_testing_data)
			cur_nb_predictions = ensemble[cur_classifier_index][4].predict_proba(np_testing_data)
			cur_weight = ensemble[cur_classifier_index][6] * relevance_for_this_data_set
			ensemble[cur_classifier_index][8] = cur_nb_predictions
			second_layer_row.append(cur_weight)
			for j in range(0,3):
				second_layer_row.append(cur_nb_predictions[j])

			# SVM Current Prediction
			cur_classifier_index = i*2 + 1
			cur_svm_predictions = ensemble[cur_classifier_index][4].predict_proba(np_testing_data)
			ensemble[cur_classifier_index][8] = cur_svm_predictions
			cur_weight = ensemble[cur_classifier_index][6] * relevance_for_this_data_set
			second_layer_row.append(cur_weight)
			for j in range(0,3):
				second_layer_row.append(cur_svm_predictions[j])

			print "set_num: ", set_num
			print "data_row: ", data_row
			print "i: ", i

		second_layer_row.append(validation_targets[data_row])
		second_layer.append(second_layer_row)
	

print second_layer.shape


'''
# predict for testing data
while 1:
	print "Please enter a sentence to be classified:"
	user_input = raw_input()
	np_testing_data = [user_input]
	print "Please enter the type of text this is. Please enter 'SMS', 'FB', 'Tweet', 'Movie Review', or 'Product Review'"
	test_type_of_text = raw_input()
	summary = []
	for i in range(0,num_of_data_sets):
		relevance_for_this_data_set = getRelevance(ensemble[i*2][1], test_type_of_text)

		# Naive Bayes Current Prediction
		cur_classifier_index = i*2
		cur_nb_predictions = ensemble[cur_classifier_index][4].predict_proba(np_testing_data)
		cur_weight = ensemble[cur_classifier_index][6] * relevance_for_this_data_set
		cur_name = ensemble[cur_classifier_index][0]
		ensemble[cur_classifier_index][8] = cur_nb_predictions
		cur_summary_data = [cur_name, cur_weight, cur_nb_predictions]
		summary.append(cur_summary_data)
	

		# SVM Current Prediction
		cur_classifier_index = i*2 + 1
		cur_svm_predictions = ensemble[cur_classifier_index][4].predict_proba(np_testing_data)
		ensemble[cur_classifier_index][8] = cur_svm_predictions
		cur_weight = ensemble[cur_classifier_index][6] * relevance_for_this_data_set
		cur_name = ensemble[cur_classifier_index][0]
		cur_summary_data = [cur_name, cur_weight, cur_svm_predictions]
		summary.append(cur_summary_data)
	
	
	print "summary for bahram ... each classifier has [name_of_classifier, weight (based on accuracy of classifier and relevance), predictions that it is in the negative, neutral, or positive class respective] ... \n= {0}".format(summary)
'''