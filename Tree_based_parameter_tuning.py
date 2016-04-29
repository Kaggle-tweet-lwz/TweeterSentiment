import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier

def main():
	accu = []
	i = 100
	while i <= 2000:
		clf = RandomForestClassifier(n_estimators = i+1, n_jobs = -1, verbose = 2, random_state = 1, max_depth = 13)
		clf.fit(X[train], y[train])
		yhat = clf.predict(X[valid])
		k = accuracy_score(y[valid], yhat)
		accu.append(k)
		i += 100
	np.savetxt("forest_estimator.csv.csv", accu)


	accu_ex = []
	i = 100
	while i <= 2000:
		clf = ExtraTreesClassifier(n_estimators = i+1, n_jobs = -1, verbose = 2, random_state = 1, max_depth = 13)
		clf.fit(X[train], y[train])
		yhat = clf.predict(X[valid])
		k = accuracy_score(y[valid], yhat)
		accu_ex.append(k)
		i += 100
	np.savetxt("ext_estimator.csv", accu_ex)

	depth_accu = []
	j = 5
	while j <= 15:
		clf = RandomForestClassifier(n_estimators = 500, n_jobs = -1, verbose = 2, random_state = 1, max_depth = j)
		clf.fit(X[train], y[train])
		yhat = clf.predict(X[valid])
		k = accuracy_score(y[valid], yhat)
		depth_accu.append(k)
		j += 1
	np.savetxt("forest_depth.csv", depth_accu)
		
	d = []
	j = 5
	while j <= 15:
		clf = ExtraTreesClassifier(n_estimators = 500, n_jobs = -1, verbose = 2, random_state = 1, max_depth = j)
		clf.fit(X[train], y[train])
		yhat = clf.predict(X[valid])
		k = accuracy_score(y[valid], yhat)
		accu.append(k)
		j +=
	np.savetxt("ext_depth.csv", d)


if __name__ == '__main__':
	main()