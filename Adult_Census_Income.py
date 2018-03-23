import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from time import time
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split



def labels(y):
    label = []
    for i in y:
        if i=='<=50K':
            label.append([0])
        else:
            label.append([1])
    return label

def convert(data):
    data = convertSex(data)
    return data

def convertSex(data):
    for i in range (len(data)):
        if data[i][4]=='Female':
            data[i][4] = 0
        else:
            data[i][4] = 1
    return data

def metric(y,pred):
    avg = 'weighted'
    print('Precision score = ',metrics.precision_score(y, pred, average=avg))
    print('Recall score = ',metrics.recall_score(y, pred, average=avg))
    print('f1-score = ',metrics.f1_score(y, pred, average=avg))


def model(x,y):
	# Train Test Split

	x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
	print(len(x_train))
	print(len(x_test))


	# DecisionTreeClassifier


	t = time()
	clf = DecisionTreeClassifier(criterion='entropy')
	print(clf)
	clf.fit(x_train,y_train)
	print('Train Time DTC = ',(time()-t))
	pre = clf.predict(x_test)
	print(pre)
	print('Accuracy = ',clf.score(x_test,y_test))

	metric(y_test,pre)
	print('Time DTC = ',(time()-t))


	# SVC


	def svm_svc(kernel,c=1.0):
	    for i in kernel:
	        print('Kernel = ',i)
	        t = time()
	        clf = SVC(kernel=i,C=c)
	        print(clf)
	        clf.fit(x_train,y_train.ravel())
	        print('Train Time SVC = ',(time()-t))
	        pre = clf.predict(x_test)
	        print(pre)
	        print('Accuracy = ',clf.score(x_test,y_test))

	        metric(y_test,pre)
	        print('Time SVC = ',(time()-t),'\n\n')

	kernel = ['linear','rbf','sigmoid','poly']
	svm_svc(kernel,c=0.1)
	print('\nC=1.0')
	svm_svc(kernel)


	# KNeighborsClassifier


	t = time()
	clf = KNeighborsClassifier()
	print(clf)
	clf.fit(x_train,y_train)
	print('Train Time DTC = ',(time()-t))
	pre = clf.predict(x_test)
	print(pre)
	print('Accuracy = ',clf.score(x_test,y_test))

	metric(y_test,pre)
	print('Time DTC = ',(time()-t))



def main():
	df = pd.read_csv('adult.csv')
	print(df.head(5))

	x = df[['age','fnlwgt','education.num','hours.per.week','sex','capital.loss']].values
	x = convert(x)

	y = np.array(labels(df[['income']].values))
	print(len(x))
	print(len(y))
	model(x,y)

	x = df[['age','education.num','hours.per.week','sex','capital.loss']].values
	x = convert(x)
	model(x,y)

if __name__ == '__main__':
	main()