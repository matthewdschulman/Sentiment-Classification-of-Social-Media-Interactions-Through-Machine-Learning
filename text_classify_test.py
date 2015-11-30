'''
    Part II (1) Text Classification and ROC
    AUTHOR Bahram Banisadr
'''

from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
import sklearn.metrics
import time
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# Retrieve training and testing datasets
newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True, remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', shuffle=True, remove=('headers', 'footers', 'quotes'))


# Create pipelines for NB and SVM classifiers
text_clf_NB = Pipeline([('vect', CountVectorizer(stop_words = 'english')),
	                     ('tfidf', TfidfTransformer(sublinear_tf=True, norm='l2')),
	                     ('clf', MultinomialNB())])

text_clf_SVM = Pipeline([('vect', CountVectorizer(stop_words = 'english')),
	                     ('tfidf', TfidfTransformer(sublinear_tf=True, norm='l2')),
	                     ('clf', svm.SVC(kernel=sklearn.metrics.pairwise.cosine_similarity, probability=True))])


# Fit NB and SVM models & time them
NB_start = time.clock()
text_clf_NB = text_clf_NB.fit(newsgroups_train.data, newsgroups_train.target)
NB_stop = time.clock()

SVM_start = time.clock()
text_clf_SVM = text_clf_SVM.fit(newsgroups_train.data, newsgroups_train.target)
SVM_stop = time.clock()


# Make predictions
NB_train_predictions = text_clf_NB.predict(newsgroups_train.data)
NB_test_predictions = text_clf_NB.predict(newsgroups_test.data)

SVM_train_predictions = text_clf_SVM.predict(newsgroups_train.data)
SVM_test_predictions = text_clf_SVM.predict(newsgroups_test.data)


# Compute metrics
NB_train_accuracy = np.mean(newsgroups_train.target == NB_train_predictions)
SVM_train_accuracy = np.mean(newsgroups_train.target == SVM_train_predictions)

NB_test_accuracy = np.mean(newsgroups_test.target == NB_test_predictions)
SVM_test_accuracy = np.mean(newsgroups_test.target == SVM_test_predictions)

NB_train_precision_recall = precision_recall_fscore_support(newsgroups_train.target, NB_train_predictions, average='binary')
NB_train_precision = NB_train_precision_recall[0]
NB_train_recall = NB_train_precision_recall[1]
SVM_train_precision_recall = precision_recall_fscore_support(newsgroups_train.target, SVM_train_predictions, average='binary')
SVM_train_precision = SVM_train_precision_recall[0]
SVM_train__recall = SVM_train_precision_recall[1]

NB_test_precision_recall = precision_recall_fscore_support(newsgroups_test.target, NB_test_predictions, average='binary')
NB_test_precision = NB_test_precision_recall[0]
NB_test_recall = NB_test_precision_recall[1]
SVM_test_precision_recall = precision_recall_fscore_support(newsgroups_test.target, SVM_test_predictions, average='binary')
SVM_test_precision = SVM_test_precision_recall[0]
SVM_test_recall = SVM_test_precision_recall[1]

NB_train_time = NB_stop - NB_start 
SVM_train_time = SVM_stop = SVM_start


# Print table of statistics
print "\n"
print "ACCURACY_TRAIN_NB: ", NB_train_accuracy
print "ACCURACY_TRAIN_SVM: ", SVM_train_accuracy
print "\n"

print "ACCURACY_TEST_NB: ", NB_test_accuracy
print "ACCURACY_TEST_SVM: ", SVM_test_accuracy
print "\n"

print "PRECISION_TRAIN_NB: ", NB_train_precision
print "PRECISION_TRAIN_SVM: ", SVM_train_precision
print "\n"

print "PRECISION_TEST_NB: ", NB_test_precision
print "PRECISION_TEST_SVM: ", SVM_test_precision
print "\n"

print "RECALL_TRAIN_NB: ", NB_train_recall
print "RECALL_TRAIN_SVM: ", SVM_train__recall
print "\n"

print "RECALL_TEST_NB: ", NB_test_recall
print "RECALL_TEST_SVM: ", SVM_test_recall
print "\n"

print "TIME_TRAIN_NB: ", NB_train_time
print "TIME_TRAIN_SVM: ", SVM_train_time
print "\n"

'''
	Plotting ROC Curve
	Referencing example found at:
	http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
'''

# Binarize the output
targets = label_binarize(newsgroups_test.target, classes = np.unique(newsgroups_test.target))

# Requested Classes
requested_classes = ['comp.graphics', 'comp.sys.mac.hardware', 'rec.motorcycles', 'sci.space', 'talk.politics.mideast']
class_indices = np.asarray([newsgroups_train.target_names.index(i) for i in requested_classes])
n_classes = class_indices.size

# Decision Functions
NB_score = text_clf_NB.predict_proba(newsgroups_test.data)
SVM_score = text_clf_SVM.predict_proba(newsgroups_test.data)

# Initialize dictionaries
NB_fpr = dict()
SVM_fpr = dict()
NB_tpr = dict()
SVM_tpr = dict()
NB_roc_auc = dict()
SVM_roc_auc = dict()

# Compute ROC curve and ROC area for each class
for i in range(n_classes):
    NB_fpr[i], NB_tpr[i], _ = roc_curve(targets[:, class_indices[i]], NB_score[:, class_indices[i]])
    NB_roc_auc[i] = auc(NB_fpr[i], NB_tpr[i])

    SVM_fpr[i], SVM_tpr[i], _ = roc_curve(targets[:, class_indices[i]], SVM_score[:, class_indices[i]])
    SVM_roc_auc[i] = auc(SVM_fpr[i], SVM_tpr[i])

# Plot of a ROC curve for a specific class
with PdfPages('graphTextClassifierROC.pdf') as plot:
	plt.figure()
	for i in range(n_classes):
		plt.plot(NB_fpr[i], NB_tpr[i], label='NB ROC curve of class {0} (area = {1:0.3f})'
	                                   ''.format(requested_classes[i], NB_roc_auc[i]))
		plt.plot(SVM_fpr[i], SVM_tpr[i], label='SVM ROC curve of class {0} (area = {1:0.3f})'
	                                   ''.format(requested_classes[i], SVM_roc_auc[i]))
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right", prop={'size':4})
	plot.savefig()
	plt.close()