
import os
import pickle
import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing


def eval(cls,k,data,target, method=None, test_ratio=0.25):
    score = 0
    if method == 'kfold':
        kfold = model_selection.KFold(n_splits=k)
        for ind_train,ind_test in kfold.split(data):
            cls.fit(data[ind_train], target[ind_train])
#             ypred = cls.predict(data[ind_test])
#             score += np.mean((ypred == target[ind_test]).astype(int))
            score += cls.score(data[ind_test],target[ind_test])
    else:
        for I in range(k):
            xtrain,xtest,ytrain,ytest=model_selection.train_test_split(data,target,test_size=test_ratio)
            cls.fit(xtrain,ytrain)
            score += cls.score(xtest,ytest)
    return score/k



df = pd.read_excel(r"diabetes.xlsx")
print(df.head())
print(df.dtypes)
print('=========================================================')

# slice the data
data = df.values[:,0:7].astype(float)
target = df.values[:,-1].astype(int)

# Evaluate some Classifiers and save the pickle file

# # LogisticRegression
# from sklearn.linear_model import LogisticRegression
# logisticReg = LogisticRegression()
# print('LogisticRegression:\t\t' ,eval(logisticReg,10,data,target,method='kfold'))
#
# # DecisionTreeClassifier
# from sklearn.tree import DecisionTreeClassifier
# decisionTree = DecisionTreeClassifier(max_depth=10)
# print('DecisionTreeClassifier: ' ,eval(decisionTree,10,data,target,method='kfold'))
#
# # RandomForestClassifier
# from sklearn import ensemble
# randomForest = ensemble.RandomForestClassifier()
# print('RandomForestClassifier: ' ,eval(randomForest,10,data,target,method='kfold'))
#
# # SVM
# from sklearn import svm
# clf = svm.SVC()
# print('Svm:\t\t\t\t\t', eval(clf, 10, data, target, method='kfold'))


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(8, 4, 2, 1), solver='sgd', max_iter=250, verbose=True)
print('MLPClassifier: ' ,eval(mlp,10,data,target,method='kfold'))


# # Save the best Pickle file
# pickle._dump(randomForest, open('diabetesRandomForest.pkl', 'wb'))
# print('\nyour pickle file save to: ' + os.getcwd())

print('=========================================================')

