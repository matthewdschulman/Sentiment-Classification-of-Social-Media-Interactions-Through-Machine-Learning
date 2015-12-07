import numpy as np
import time
import sklearn.metrics
import matplotlib.pyplot as plt
import sys

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



targets = np.array(['-1', '1', '-1'])

predictions = np.array(['-1', '1', '1'])

targets =  np.append(targets,'0')
predictions =  np.append(predictions,'0')

print "targets: ", targets
print "predictions: ", predictions

print "hello world"
print "Scores: ", precision_recall_fscore_support(targets, predictions,average='binary')
