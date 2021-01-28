from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings
from sklearn.datasets import load_iris

warnings.filterwarnings('ignore')

dataset = load_iris()
x= dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=32)

allAlgorithms = all_estimators(type_filter='classifier')

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률 :', accuracy_score(y_test, y_pred))
    except:
        # continue
        print(name,'은 없는 놈!')

import sklearn
print(sklearn.__version__) # 0.23.2

# AdaBoostClassifier 의 정답률 : 0.9666666666666667
# BaggingClassifier 의 정답률 : 1.0
# BernoulliNB 의 정답률 : 0.3
# CalibratedClassifierCV 의 정답률 : 0.8666666666666667
# CategoricalNB 의 정답률 : 0.9333333333333333
# CheckingClassifier 의 정답률 : 0.4
# ClassifierChain 은 없는 놈!
# ComplementNB 의 정답률 : 0.7
# DecisionTreeClassifier 의 정답률 : 1.0
# DummyClassifier 의 정답률 : 0.3333333333333333
# ExtraTreeClassifier 의 정답률 : 0.8666666666666667
# ExtraTreesClassifier 의 정답률 : 1.0
# GaussianNB 의 정답률 : 0.9666666666666667
# GaussianProcessClassifier 의 정답률 : 0.9666666666666667
# GradientBoostingClassifier 의 정답률 : 1.0
# HistGradientBoostingClassifier 의 정답률 : 0.9666666666666667
# KNeighborsClassifier 의 정답률 : 1.0
# LabelPropagation 의 정답률 : 1.0
# LabelSpreading 의 정답률 : 1.0
# LinearDiscriminantAnalysis 의 정답률 : 1.0
# LinearSVC 의 정답률 : 1.0
# LogisticRegression 의 정답률 : 1.0
# LogisticRegressionCV 의 정답률 : 1.0
# MLPClassifier 의 정답률 : 1.0
# MultiOutputClassifier 은 없는 놈!
# MultinomialNB 의 정답률 : 0.9666666666666667
# NearestCentroid 의 정답률 : 0.9333333333333333
# NuSVC 의 정답률 : 0.9666666666666667
# OneVsOneClassifier 은 없는 놈!
# OneVsRestClassifier 은 없는 놈!
# OutputCodeClassifier 은 없는 놈!
# PassiveAggressiveClassifier 의 정답률 : 0.8
# Perceptron 의 정답률 : 0.8
# QuadraticDiscriminantAnalysis 의 정답률 : 1.0
# RadiusNeighborsClassifier 의 정답률 : 0.9666666666666667
# RandomForestClassifier 의 정답률 : 0.9666666666666667
# RidgeClassifier 의 정답률 : 0.8666666666666667
# RidgeClassifierCV 의 정답률 : 0.8666666666666667
# SGDClassifier 의 정답률 : 0.8666666666666667
# SVC 의 정답률 : 0.9666666666666667
# StackingClassifier 은 없는 놈!
# VotingClassifier 은 없는 놈!