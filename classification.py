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
	elif x == 'neutral' or x == 'objective' or x == 'objective-OR-neutral':
		return 0
	else:
		print "ERRor WITH MAPPING TO NUMERIC VALS"
		return -5

def mapProductReviewData(x):
	if x == '0':
		return -1
	elif x == '1':
		return 1
	else: 
		return 0

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
	#sorted = [dataType, testType]
	#sorted.sort()
	#firstType = sorted[0]
	#secondType = sorted[1]

	
	print "dataType = {0} and testType = {1}".format(dataType, testType)
	if dataType == testType:
		return 1.0
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
		elif dataType == 'Tweet':
			return 0.6
		else: 
			return 0.9 # product <> movie
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

# twitter-A training data
# with open('data/twitter/discrete/twitter-A-downloaded.tsv', 'r') as f:
# 	cur_training = [x.strip().split('\t') for x in f]
# cur_np_training = np.array(cur_training)
# cur_np_training_data = cur_np_training[:,5]
# cur_np_training_target = map(mapToNumericTargetValues, cur_np_training[:,4])
# ensemble = appendTrainingDataToEnsemble(ensemble, 'Twitter-A Naive Bayes', 'Tweet', cur_np_training_data, cur_np_training_target)
# ensemble = appendTrainingDataToEnsemble(ensemble, 'Twitter-A SVM', 'Tweet', cur_np_training_data, cur_np_training_target)
# num_of_data_sets += 1
# 
# # twitter-B training data
# with open('data/twitter/discrete/twitter-B-downloaded.tsv', 'r') as f:
# 	cur_training = [x.strip().split('\t') for x in f]
# cur_np_training = np.array(cur_training)
# cur_np_training_data = cur_np_training[:,3]
# cur_np_training_target = map(mapToNumericTargetValues, cur_np_training[:,2])
# ensemble = appendTrainingDataToEnsemble(ensemble, 'Twitter-B Naive Bayes', 'Tweet', cur_np_training_data, cur_np_training_target)
# ensemble = appendTrainingDataToEnsemble(ensemble, 'Twitter-B SVM', 'Tweet', cur_np_training_data, cur_np_training_target)
# num_of_data_sets += 1

# product-review-A training data
cur_training = []
with open('data/product_review/sentiment-labelled-sentences/amazon_cells_labelled.txt','r') as f:
	for x in f:
		cur_line_arr = x.strip().split('\t')
		if len(cur_line_arr) > 1:
			cur_training.append(cur_line_arr)
cur_np_training = np.array(cur_training)
cur_np_training_data = cur_np_training[:,0]
cur_np_training_target = cur_np_training[:,1]
cur_np_training_target = map(mapProductReviewData, cur_np_training[:,1])
ensemble = appendTrainingDataToEnsemble(ensemble, 'Product-Review-A Naive Bayes', 'Product Review', cur_np_training_data, cur_np_training_target)
ensemble = appendTrainingDataToEnsemble(ensemble, 'Product-Review-A SVM', 'Product Review', cur_np_training_data, cur_np_training_target)
num_of_data_sets += 1

# product-review-B training data
cur_training = []
with open('data/product_review/sentiment-labelled-sentences/yelp_labelled.txt','r') as f:
	for x in f:
		cur_line_arr = x.strip().split('\t')
		if len(cur_line_arr) > 1:
			cur_training.append(cur_line_arr)
cur_np_training = np.array(cur_training)
cur_np_training_data = cur_np_training[:,0]
cur_np_training_target = cur_np_training[:,1]
cur_np_training_target = map(mapProductReviewData, cur_np_training[:,1])
ensemble = appendTrainingDataToEnsemble(ensemble, 'Product-Review-B Naive Bayes', 'Product Review', cur_np_training_data, cur_np_training_target)
ensemble = appendTrainingDataToEnsemble(ensemble, 'Product-Review-B SVM', 'Product Review', cur_np_training_data, cur_np_training_target)
num_of_data_sets += 1

# movie-review training data
cur_training = []
with open('data/product_review/sentiment-labelled-sentences/imdb_labelled.txt','r') as f:
	for x in f:
		cur_line_arr = x.strip().split('\t')
		if len(cur_line_arr) > 1:
			cur_training.append(cur_line_arr)
cur_np_training = np.array(cur_training)
cur_np_training_data = cur_np_training[:,0]
cur_np_training_target = cur_np_training[:,1]
cur_np_training_target = map(mapProductReviewData, cur_np_training[:,1])
ensemble = appendTrainingDataToEnsemble(ensemble, 'Movie-Review Naive Bayes', 'Movie Review', cur_np_training_data, cur_np_training_target)
ensemble = appendTrainingDataToEnsemble(ensemble, 'Movie-Review SVM', 'Movie Review', cur_np_training_data, cur_np_training_target)
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

# set up relevance scores
# relevance = np.zeros((4,4))
# for i in range(0,4):
# 	cur_accuracy = -1.0
# 	if i == 0:
# 		# movie review
# 		cur_accuracy = 1.0 #TODO: Implement when we have movie review datasets
# 	elif i == 1:
# 		# product review
# 		cur_accuracy = 1.0 #TODO: Implement when we have product review datasets
# 	elif i == 2:
# 		# sms/fb
# 	 	cur_accuracy = np.average(ensemble[0][6], ensemble[1][6], ensemble[2][6], ensemble[3][6])	
# 	elif i == 3:
# 		# tweet
# 		cur_accuracy = 1.0 #TODO: Implement when we have product review datasets
# 
# 	for j in range(0,4):
# 		if i == j:
# 			relevance[i][j] = 1.0
# 		if i < j: #Only fill on half of the table
# 			if j == 1:
# 				# testing on movie
# 				relevance[i][j] = 1.0

		


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
		print "relevance = {0}".format(relevance_for_this_data_set)

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
