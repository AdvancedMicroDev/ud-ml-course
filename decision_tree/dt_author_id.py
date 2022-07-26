#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
import sys
sys.path.insert(0, "D:\\ud120-projects\\tools\\")
from sklearn.metrics import accuracy_score
from email_preprocess import preprocess
from time import time
from sklearn import tree


# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
print(len(features_train[0]))
clf = tree.DecisionTreeClassifier(min_samples_split=40)

t0 = time()
clf.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")

acc = accuracy_score(pred, labels_test) * 100
print(f"Final Accuracy: {acc:.4f}%")
#########################################################