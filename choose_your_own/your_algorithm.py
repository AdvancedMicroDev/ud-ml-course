#!/usr/bin/python


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

features_train, labels_train, features_test, labels_test = makeTerrainData()


# the training data (features_train, labels_train) have both "fast" and "slow"
# points mixed together--separate them so we can give them different colors
# in the scatterplot and identify them visually
grade_fast = [features_train[ii][0]
              for ii in range(0, len(features_train)) if labels_train[ii] == 0]
bumpy_fast = [features_train[ii][1]
              for ii in range(0, len(features_train)) if labels_train[ii] == 0]
grade_slow = [features_train[ii][0]
              for ii in range(0, len(features_train)) if labels_train[ii] == 1]
bumpy_slow = [features_train[ii][1]
              for ii in range(0, len(features_train)) if labels_train[ii] == 1]


# initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color="b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color="r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


# your code here!  name your classifier object clf if you want the
# visualization code (prettyPicture) to show you the decision boundary
svc = SVC(probability=True)
clf = AdaBoostClassifier(n_estimators=5, base_estimator=svc)
clf_knn = KNeighborsClassifier()
clf_rf = RandomForestClassifier(max_depth=8, n_estimators=10)
parameters = {
    "n_estimators": [5, 10, 50, 100, 250]
    # "max_depth": [2, 4, 8, 16, 32, None]
}

cv = GridSearchCV(clf, param_grid=parameters, cv=5)
cv.fit(features_train, labels_train)

clf.fit(features_train, labels_train)
clf_knn.fit(features_train, labels_train)
clf_rf.fit(features_train, labels_train)

pred = clf.predict(features_test)
pred_knn = clf_knn.predict(features_test)
pred_rf = clf_rf.predict(features_test)

acc = accuracy_score(y_true=labels_test, y_pred=pred) * 100
acc_knn = accuracy_score(y_true=labels_test, y_pred=pred_knn) * 100
acc_rf = accuracy_score(y_true=labels_test, y_pred=pred_rf) * 100

print(f"Accuracy of Adaboost is: {acc:.4f}%")
print(f"Accuracy of kNN is: {acc_knn:.4f}%")
print(f"Accuracy of Random FOrest is: {acc_rf:.4f}%")


def display(results):
    print(f'Best parameters are: {results.best_params_}')
    print("\n")
    mean_score = results.cv_results_['mean_test_score']
    std_score = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    for mean, std, params in zip(mean_score, std_score, params):
        print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')


display(cv)

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
