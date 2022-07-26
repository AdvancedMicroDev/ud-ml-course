#!/usr/bin/python3
import sys
sys.path.insert(0, "D:\\ud120-projects\\tools\\")

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
from time import time
from sklearn.metrics import accuracy_score
from email_preprocess import preprocess
from sklearn import svm


# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
clf = svm.SVC(kernel="rbf", C=10000.0)

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

t0 = time()
clf.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")

acc = accuracy_score(pred, labels_test) * 100
print(f"Final Accuracy: {acc:.4f}%")

# ans_10, ans_26, ans_50 = pred[10], pred[26], pred[50]
# print(f"Ans for 10: {ans_10}, Ans for 26: {ans_26}, Ans for 50: {ans_50}")

print("No of events pred as Chris: ", sum(clf.predict(features_test) == 1))
# count = 0
# for i in range(len(pred)):
#     if pred[i]:
#         count += 1

#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''


#########################################################
