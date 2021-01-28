from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings
from sklearn.datasets import load_breast_cancer

warnings.filterwarnings('ignore')

dataset = load_breast_cancer()
x= dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=32)
kfold = KFold(n_splits=5, shuffle=True)

allAlgorithms = all_estimators(type_filter='classifier')

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        score = cross_val_score(model, x_train, y_train, cv=kfold)
        print(name, '의 정답률 :\n', score)
    except:
        # continue
        print(name,'은 없는 놈!')

import sklearn
print(sklearn.__version__) # 0.23.2

# AdaBoostClassifier 의 정답률 :
#  [0.95604396 0.95604396 0.95604396 0.95604396 0.95604396]
# BaggingClassifier 의 정답률 :
#  [0.87912088 0.91208791 0.94505495 0.95604396 0.93406593]
# BernoulliNB 의 정답률 :
#  [0.61538462 0.67032967 0.6043956  0.69230769 0.57142857]
# CalibratedClassifierCV 의 정답률 :
#  [0.94505495 0.91208791 0.91208791 0.91208791 0.95604396]
# CategoricalNB 은 없는 놈!
# CheckingClassifier 의 정답률 :
#  [0. 0. 0. 0. 0.]
# ClassifierChain 은 없는 놈!
# ComplementNB 의 정답률 :
#  [0.9010989  0.91208791 0.91208791 0.9010989  0.92307692]
# DecisionTreeClassifier 의 정답률 :
#  [0.94505495 0.86813187 0.97802198 0.92307692 0.9010989 ]
# DummyClassifier 의 정답률 :
#  [0.50549451 0.56043956 0.56043956 0.56043956 0.53846154]
# ExtraTreeClassifier 의 정답률 :
#  [0.93406593 0.94505495 0.87912088 0.92307692 0.87912088]
# ExtraTreesClassifier 의 정답률 :
#  [0.98901099 0.95604396 0.94505495 0.96703297 0.93406593]
# GaussianNB 의 정답률 :
#  [0.95604396 0.94505495 0.93406593 0.94505495 0.94505495]
# GaussianProcessClassifier 의 정답률 :
#  [0.96703297 0.92307692 0.89010989 0.89010989 0.94505495]
# GradientBoostingClassifier 의 정답률 :
#  [0.94505495 0.95604396 0.94505495 0.96703297 0.96703297]
# HistGradientBoostingClassifier 의 정답률 :
#  [0.97802198 0.95604396 0.95604396 0.94505495 1.        ]
# KNeighborsClassifier 의 정답률 :
#  [0.94505495 0.9010989  0.93406593 0.95604396 0.95604396]
# LabelPropagation 의 정답률 :
#  [0.38461538 0.41758242 0.42857143 0.36263736 0.36263736]
# LabelSpreading 의 정답률 :
#  [0.38461538 0.37362637 0.41758242 0.3956044  0.40659341]
# LinearDiscriminantAnalysis 의 정답률 :
#  [0.96703297 0.89010989 0.87912088 0.97802198 0.98901099]
# LinearSVC 의 정답률 :
#  [0.95604396 0.95604396 0.93406593 0.94505495 0.9010989 ]
# LogisticRegression 의 정답률 :
#  [0.95604396 0.95604396 0.95604396 0.95604396 0.91208791]
# LogisticRegressionCV 의 정답률 :
#  [0.96703297 0.95604396 0.93406593 0.98901099 0.92307692]
# MLPClassifier 의 정답률 :
#  [0.9010989  0.94505495 0.95604396 0.93406593 0.91208791]
# MultiOutputClassifier 은 없는 놈!
# MultinomialNB 의 정답률 :
#  [0.92307692 0.87912088 0.89010989 0.9010989  0.95604396]
# NearestCentroid 의 정답률 :
#  [0.87912088 0.91208791 0.87912088 0.94505495 0.87912088]
# NuSVC 의 정답률 :
#  [0.86813187 0.92307692 0.84615385 0.9010989  0.87912088]
# OneVsOneClassifier 은 없는 놈!
# OneVsRestClassifier 은 없는 놈!
# OutputCodeClassifier 은 없는 놈!
# PassiveAggressiveClassifier 의 정답률 :
#  [0.92307692 0.91208791 0.94505495 0.92307692 0.87912088]
# Perceptron 의 정답률 :
#  [0.91208791 0.87912088 0.78021978 0.91208791 0.87912088]
# QuadraticDiscriminantAnalysis 의 정답률 :
#  [0.95604396 0.94505495 0.93406593 0.96703297 0.96703297]
# RadiusNeighborsClassifier 은 없는 놈!
# RandomForestClassifier 의 정답률 :
#  [0.95604396 0.97802198 0.97802198 0.95604396 0.91208791]
# RidgeClassifier 의 정답률 :
#  [0.97802198 0.94505495 0.94505495 0.95604396 0.95604396]
# RidgeClassifierCV 의 정답률 :
#  [0.95604396 0.96703297 0.93406593 0.96703297 0.91208791]
# SGDClassifier 의 정답률 :
#  [0.9010989  0.94505495 0.89010989 0.93406593 0.89010989]
# SVC 의 정답률 :
#  [0.91208791 0.91208791 0.92307692 0.95604396 0.95604396]
# StackingClassifier 은 없는 놈!
# VotingClassifier 은 없는 놈!