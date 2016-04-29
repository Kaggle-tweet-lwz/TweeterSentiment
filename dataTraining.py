import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
from sklearn.metrics import classification_report


def randomForest(X, y, train, valid):
	clf = RandomForestClassifier(n_estimators = 500, n_jobs = -1, verbose = 2, random_state = 1, max_depth = 13)
	clf.fit(X[train], y[train])
	yhat = clf.predict(X[valid])
	accuracy_score(y[valid], yhat)
	print("randomForest" + str(accuracy_score(y[valid], yhat)))
	yhat_prob = clf.predict_proba(X[valid])[:,1]
	print(classification_report(y[valid], yhat))
	print("randomForest roc_accuracy" + str(roc_auc_score(y[valid], yhat_prob)))
	np.savetxt("y_rf.csv", yhat_prob)
	return yhat_prob

def gradientBoost(X, y, train, valid):
	from sklearn.ensemble import GradientBoostingClassifier
	clf1 = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=1, random_state=0).fit(X[train], y[train])
	print("gradientboosting" + str(clf1.score(X[valid].toarray(), y[valid])))
	yhat = clf1.predict(X[valid].toarray())
	yhat_prob = clf1.predict_proba(X[valid].toarray())[:,1]
	print(classification_report(y[valid], yhat))
	print("gradient boosting roc_accuracy" + str(roc_auc_score(y[valid], yhat_prob)))
	np.savetxt("y_gb.csv", yhat_prob)
	return yhat_prob

def extraTree(X, y, train, valid):
	clf = ExtraTreesClassifier(n_jobs = -1, n_estimators = 300, verbose = 2,
            random_state = 1, max_depth = 10, bootstrap = True)
	clf.fit(X[train], y[train])
	yhat = clf.predict(X[valid])
	yhat_prob = clf.predict_proba(X[valid])[:,1]
	print("extra tree randomForest" + str(accuracy_score(y[valid], yhat)))
	print(classification_report(y[valid], yhat))

	print("extra tree randomForest roc_accuracy" + str(roc_auc_score(y[valid], yhat_prob)))
	np.savetxt("y_extratree.csv", yhat_prob)
	return yhat_prob

def adaboost(X, y, train, valid):
	from sklearn.ensemble import AdaBoostClassifier
	clf2 = AdaBoostClassifier(n_estimators=100).fit(X[train], y[train])
	yhat = clf2.predict(X[valid])
	print(classification_report(y[valid], yhat))
	accuracy_score(y[valid], yhat)
	print("adaboost" + str(accuracy_score(y[valid], yhat)))
	yhat_prob = clf2.predict_proba(X[valid])[:,1]
	print("extra tree randomForest roc_accuracy" + str(roc_auc_score(y[valid], yhat_prob)))
	np.savetxt("y_ada.csv", yhat_prob)
	return yhat_prob

def logisticRegressionRidge(X, y, train, valid):
	from sklearn import linear_model
	model = linear_model.SGDClassifier(loss='log', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=True, verbose=0, epsilon=0.1, n_jobs=-1, random_state=None, 
	learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, average=False).fit(X[train], y[train])
	yhat = model.predict(X[valid])
	accuracy_score(y[valid], yhat)
	print(classification_report(y[valid], yhat))

	print("logistic regression and ridge accuracy" + str(accuracy_score(y[valid], yhat)))
	yhat_prob = model.predict_proba(X[valid])[:,1]
	roc_auc_score(y[valid], yhat_prob)
	print("logistic regression and ridge roc_accuracy" + str(roc_auc_score(y[valid], yhat_prob)))
	np.savetxt("y_log.csv", yhat_prob)
	return yhat_prob

def linearSVM(X, y, train, valid):
	from sklearn import linear_model
	model = linear_model.SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=True, verbose=0, epsilon=0.1, n_jobs=-1, random_state=None, 
	learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, average=False).fit(X[train], y[train])
	yhat = model.predict(X[valid])
	print(classification_report(y[valid], yhat))
	accuracy_score(y[valid], yhat)
	print("linear SVM accuracy" + str(accuracy_score(y[valid], yhat)))
	yhat_prob = model.predict_proba(X[valid])[:,1]
	roc_auc_score(y[valid], yhat_prob)
	print("linear SVM roc_accuracy" + str(roc_auc_score(y[valid], yhat_prob)))
	np.savetxt("y_svm.csv", yhat)
	# try to run kernel, runtime too long
	# from sklearn.svm import SVC
	# clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape=None, degree=3, gamma='auto', kernel='linear',max_iter=-1, probability=True, random_state=None, shrinking=True,tol=0.001, verbose=False)
	# clf.fit(X[train], y[train])
 #    yhat_prob = clf.predict_proba(X[valid])[:,1]
	return yhat

def logisticRegressionLasso(X, y, train, valid):
	from sklearn import linear_model
	model = linear_model.SGDClassifier(loss='log', penalty='l1', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=True, verbose=0, epsilon=0.1, n_jobs=-1, random_state=None, 
	learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, average=False).fit(X[train], y[train])
	yhat = model.predict(X[valid])
	accuracy_score(y[valid], yhat)
	print(classification_report(y[valid], yhat))

	print("logistic regression and lasso accuracy" + str(accuracy_score(y[valid], yhat)))
	yhat_prob = model.predict_proba(X[valid])[:,1]
	roc_auc_score(y[valid], yhat_prob)
	print("logistic regression and lasso roc_accuracy" + str(roc_auc_score(y[valid], yhat_prob)))
	np.savetxt("y_loglassp.csv", yhat_prob)
	return yhat_prob

def main():

	df = np.load("temp2.npz") #file in corresponding directory
	y = df['y']
	X = df["X"][()]
	X = np.load("newX.npz")["X"][()]

	# try to apply weights 
	# weight = [math.log(1+1578627/i) for i in n]
	# weight = np.array(weight)
	# weight = weight.astype(np.uint8)
	# for i in range(5000):
	# 	X[:,i] *=weight[i]


	index = []
	i = []
	for k in range(len(y)):
		if y[k]!=-1:
			index.append(k)
		i.append(k)

	# train = index
	# valid = range(50000)

	#normal training 
	np.random.shuffle(index)
	train = index[:int(len(y) * .6)]
	valid = index[int(len(y) * .6):]



	randomForest(X, y, train, valid)
	gradientBoost(X, y, train, valid)
	extraTree(X, y, train, valid)
	adaboost(X, y, train, valid)
	logisticRegressionRidge(X, y, train, valid)
	logisticRegressionLasso(X, y, train, valid)
	linearSVM(X, y, train, valid)
	np.savetxt("y_data.csv", y[valid])


	

if __name__ == '__main__':
 	main()