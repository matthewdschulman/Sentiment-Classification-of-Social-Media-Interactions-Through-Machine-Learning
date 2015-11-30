'''
    Text Sentiment Classification
    AUTHOR Matt Schulman and Bahram Banisadr
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
		print "ERROR WITH MAPPING TO NUMERIC VALS"
		return -5

# import the sms-test-gold-A.tsv training data
with open('data/sms/sms-test-gold-A.tsv','r') as f:
	training=[x.strip().split('\t') for x in f]

np_training = np.array(training)

# get the training data
np_training_data = np_training[:,5]

# get the training target values
np_training_target = map(mapToNumericTargetValues, np_training[:,4])

# Naive Bayes Classifier and fitting
text_clf_bayes = Pipeline([('vect', CountVectorizer(lowercase=True, stop_words='english')),
                           ('tfidf', TfidfTransformer(norm='l2', sublinear_tf=True)),
		           ('clf', MultinomialNB()),
		          ]
			 )
text_clf_bayes = text_clf_bayes.fit(np_training_data, np_training_target)

# Compute Naive Bayes predictions
training_predicted_nb = text_clf_bayes.predict(np_training_data)
training_accuracy_nb = np.mean(training_predicted_nb == np_training_target)
print "The Training Accuracy for the Naive Bayes Classifier = {0}".format(training_accuracy_nb)

# SVM Cosine Similarity Classifier and Fitting
text_clf_svm = Pipeline([('vect', CountVectorizer(lowercase=True, stop_words='english')),
 			 ('tfidf', TfidfTransformer(norm='l2', sublinear_tf=True)),
 			 ('clf', svm.SVC(kernel=sklearn.metrics.pairwise.linear_kernel, probability=True)), 
 		        ]
 		       )
text_clf_svm = text_clf_svm.fit(np_training_data, np_training_target)

# Compute SVM predictions
training_predicted_svm = text_clf_svm.predict(np_training_data)
training_accuracy_svm = np.mean(training_predicted_svm == np_training_target)
print "The Training Accuracy for the SVM Classifier = {0}".format(training_accuracy_svm)

# predict for testing data
while 1:
	print "Please enter a sentence to be classified:"
	user_input = raw_input()
	np_testing_data = [user_input]
	testing_predictions_nb = text_clf_bayes.predict(np_testing_data)
	testing_predictions_svm = text_clf_svm.predict(np_testing_data)

	# Print prediction for Naive Bayes
	if testing_predictions_nb[0] == 1:
		print "Naive Bayes predicts that your sentence is positive"
	elif testing_predictions_nb[0] == -1:
		print "Naive Bayes predicts that your sentence is negative"
	elif testing_predictions_nb[0] == 0:
		print "Naive Bayes predicts that your sentence is neutral"

	# Print prediction for SVM
	if testing_predictions_svm[0] == 1:
		print "SVM predicts that your sentence is positive\n"
	elif testing_predictions_svm[0] == -1:
		print "SVM Bayes predicts that your sentence is negative\n"
	elif testing_predictions_svm[0] == 0:
		print "SVM Bayes predicts that your sentence is neutral\n"
