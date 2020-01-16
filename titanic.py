#importing the packages
import numpy as np
import pandas as pd
# data visualization
import seaborn as sns

#Algorithms
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

# getting the data
train = pd.read_csv("titanic_train.csv")
test = pd.read_csv("titanic_test.csv")
print(train)
#train.info()
#print(train.describe())
#mac=train.isnull().sum()
#per=mac/len(train)
#print(per)
Age_median = train['Age'].mean()
train['Age'].fillna(Age_median,inplace = True)
#print(train['Age'])
#print(train.isnull().sum())
common_value = 'S'
train.fillna(common_value,inplace = True)
print(train.isnull().sum())
train.drop("Cabin",axis = 1,inplace = True)
print(train.isnull().sum())

# analysing the data is done in jupyter notebook


#Now conversion into categorical
sex=pd.get_dummies(train['Sex'],drop_first = True)
print(sex)
embar = pd.get_dummies(train['Embarked'],drop_first = True)
print(embar)
pclass = pd.get_dummies(train['Pclass'],drop_first = True)
print(pclass)
train = pd.concat([train,sex,embar,pclass],axis = 1)
print(train)
train.drop(['Sex','Embarked','PassengerId','Name','Ticket','Pclass'],axis = 1,inplace = True)
print(train)

# implementation of regression
x = train.drop("Survived",axis = 1)
y = train["Survived"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state = 0)
# Standardization of data(scaling the data)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#print(X_train)

#Gaussian naive bayes prediction
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
pred = gaussian.predict(X_test)
print(pred)
acc_gaussian = round(gaussian.score(X_train, y_train) * 100,2)
print(acc_gaussian)
# checking whether the prediction is correct or not by using the predicted values and actual values
# using confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,pred)
print(confusion_matrix)
# calculate the accuracy
"""from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,pred)
print(accuracy)"""

#ROC curve
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, pred)
roc_auc = auc(fpr, tpr)
#print "Area under the ROC curve : %f" % roc_auc
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# Logistic regression
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
pred1 = logmodel.predict(X_test)
acc_log = round(logmodel.score(X_train, y_train) * 100, 2)
print(acc_log)
print(pred1)
from sklearn.metrics import confusion_matrix
confusion_matrix1 = confusion_matrix(y_test,pred1)
print(confusion_matrix1)
# calculate the accuracy
"""from sklearn.metrics import accuracy_score
accuracy1=accuracy_score(y_test,pred1)
print(accuracy1)"""

#ROC CURVE
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, pred1)
roc_auc = auc(fpr, tpr)
#print "Area under the ROC curve : %f" % roc_auc
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


#RANDOM FOREST

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
pred2 = random_forest.predict(X_test)
print(pred2)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
print(acc_random_forest)
#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix2 = confusion_matrix(y_test,pred2)
print(confusion_matrix2)
# calculate the accuracy
"""from sklearn.metrics import accuracy_score
accuracy2=accuracy_score(y_test,pred2)
print(accuracy2)"""

#ROC curve
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, pred2)
roc_auc = auc(fpr, tpr)
#print "Area under the ROC curve : %f" % roc_auc
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# stochastic gradent descent (sgd)


sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, y_train)
pred3 = sgd.predict(X_test)
print(pred3)
acc_sgd = round(sgd.score(X_train, y_train) * 100, 2)
print(acc_sgd)
#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix3 = confusion_matrix(y_test,pred3)
print(confusion_matrix3)
# calculate the accuracy
"""from sklearn.metrics import accuracy_score
accuracy3=accuracy_score(y_test,pred3)
print(accuracy3)"""

#ROC curve
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, pred3)
roc_auc = auc(fpr, tpr)
#print "Area under the ROC curve : %f" % roc_auc
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# K NEAREST NEIGHBOUR

KNN = KNeighborsClassifier(n_neighbors = 3)
KNN.fit(X_train, y_train) 
pred4 = KNN.predict(X_test)
print(pred4) 
acc_knn = round(KNN.score(X_train, y_train) * 100, 2)
print(acc_knn)
#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix4 = confusion_matrix(y_test,pred4)
print(confusion_matrix4)
# calculate the accuracy
"""from sklearn.metrics import accuracy_score
accuracy4=accuracy_score(y_test,pred4)
print(accuracy4)"""

#ROC curve
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, pred4)
roc_auc = auc(fpr, tpr)
#print "Area under the ROC curve : %f" % roc_auc
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


#DESCISION TREE

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train,y_train)
pred5 = decision_tree.predict(X_test) 
print(pred5)
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
print(acc_decision_tree)
#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix5 = confusion_matrix(y_test,pred5)
print(confusion_matrix5)
# calculate the accuracy
"""from sklearn.metrics import accuracy_score
accuracy5=accuracy_score(y_test,pred4)
print(accuracy5)"""

#ROC curve
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, pred5)
roc_auc = auc(fpr, tpr)
#print "Area under the ROC curve : %f" % roc_auc
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

#LINEAR SUPPORT VECTOR MACHINE

linear_svc = LinearSVC()
linear_svc.fit(X_train, y_train)

pred6 = linear_svc.predict(X_test)
print(pred6)
acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)
print(acc_linear_svc)
#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix6 = confusion_matrix(y_test,pred6)
print(confusion_matrix6)
# calculate the accuracy
"""from sklearn.metrics import accuracy_score
accuracy6=accuracy_score(y_test,pred6)
print(accuracy6)"""

#ROC curve
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, pred6)
roc_auc = auc(fpr, tpr)
#print "Area under the ROC curve : %f" % roc_auc
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# PERCEPTRON

perceptron = Perceptron(max_iter=5)
perceptron.fit(X_train, y_train)
pred7 = perceptron.predict(X_test)
print(pred7)
acc_perceptron = round(perceptron.score(X_train, y_train) * 100, 2)
print(acc_perceptron)
#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix7 = confusion_matrix(y_test,pred7)
print(confusion_matrix7)
# calculate the accuracy
"""from sklearn.metrics import accuracy_score
accuracy7=accuracy_score(y_test,pred7)
print(accuracy7)"""

#ROC curve
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, pred7)
roc_auc = auc(fpr, tpr)
#print "Area under the ROC curve : %f" % roc_auc
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# FINDING WHICH IS THE BEST ACCURACY
results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 
              'Decision Tree'],
    'Score': [acc_linear_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_decision_tree]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
print(result_df.head(9))

