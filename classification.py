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
		reutnr -5
		
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
time_before_nb_training = time.clock()
text_clf_bayes = text_clf_bayes.fit(np_training_data, np_training_target)
time_after_nb_training = time.clock()

# Compute Naive Bayes predictions
training_predicted = text_clf_bayes.predict(np_training_data)
training_accuracy = np.mean(training_predicted == np_training_target)
print "The Training Accuracy = {0}".format(training_accuracy)

