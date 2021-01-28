from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings
from sklearn.datasets import load_wine

warnings.filterwarnings('ignore')

dataset = load_wine()
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
# SVC 의 정답률 :
#  [0.91208791 0.91208791 0.92307692 0.95604396 0.95604396]
# StackingClassifier 은 없는 놈!
# VotingClassifier 은 없는 놈!
# 0.23.2
# PS C:\Study>  c:; cd 'c:\Study'; & 'C:\Users\ai\Anaconda3\python.exe' 'c:\Users\ai\.vscode\extensions\ms-python.python-2021.1.502429796\pythonFiles\lib\python\debugpy\launcher' '63491' '--' 'c:\Study\ml\m11_kfold_estimators3_wine.py' 
# C:\Users\ai\Anaconda3\lib\site-packages\sklearn\utils\deprecation.py:143: FutureWarning: The sklearn.utils.testing module is  deprecated in 
# version 0.22 and will be removed in version 0.24. The corresponding classes / functions should instead be imported from sklearn.utils. Anything that cannot be imported from sklearn.utils is now part of the private API.
#   warnings.warn(message, FutureWarning)
# AdaBoostClassifier 의 정답률 :
#  [0.96551724 0.93103448 0.89285714 0.89285714 0.78571429]
# BaggingClassifier 의 정답률 :
#  [1.         0.96551724 0.96428571 0.89285714 1.        ]
# BernoulliNB 의 정답률 :
#  [0.31034483 0.4137931  0.64285714 0.32142857 0.46428571]
# CalibratedClassifierCV 의 정답률 :
#  [0.82758621 0.93103448 0.96428571 0.89285714 0.96428571]
# CategoricalNB 은 없는 놈!
# CheckingClassifier 의 정답률 :
#  [0. 0. 0. 0. 0.]
# ClassifierChain 은 없는 놈!
# ComplementNB 의 정답률 :
#  [0.75862069 0.65517241 0.57142857 0.67857143 0.60714286]
# DecisionTreeClassifier 의 정답률 :
#  [0.96551724 0.89655172 0.85714286 0.82142857 0.78571429]
# DummyClassifier 의 정답률 :
#  [0.31034483 0.48275862 0.28571429 0.35714286 0.39285714]
# ExtraTreeClassifier 의 정답률 :
#  [0.93103448 0.82758621 0.89285714 0.85714286 0.96428571]
# ExtraTreesClassifier 의 정답률 :
#  [0.96551724 1.         1.         1.         1.        ]
# GaussianNB 의 정답률 :
#  [0.96551724 0.89655172 0.96428571 0.96428571 1.        ]
# GaussianProcessClassifier 의 정답률 :
#  [0.48275862 0.48275862 0.53571429 0.28571429 0.5       ]
# GradientBoostingClassifier 의 정답률 :
#  [0.93103448 0.93103448 0.89285714 0.92857143 1.        ]
# HistGradientBoostingClassifier 의 정답률 :
#  [1.         0.96551724 1.         0.96428571 0.92857143]
# KNeighborsClassifier 의 정답률 :
#  [0.65517241 0.65517241 0.64285714 0.78571429 0.53571429]
# LabelPropagation 의 정답률 :
#  [0.5862069  0.4137931  0.35714286 0.5        0.25      ]
# LabelSpreading 의 정답률 :
#  [0.4137931  0.48275862 0.35714286 0.5        0.42857143]
# LinearDiscriminantAnalysis 의 정답률 :
#  [1.         1.         0.96428571 1.         1.        ]
# LinearSVC 의 정답률 :
#  [0.62068966 0.89655172 0.78571429 0.89285714 0.78571429]
# LogisticRegression 의 정답률 :
#  [1.         0.93103448 0.96428571 0.92857143 0.96428571]
# LogisticRegressionCV 의 정답률 :
#  [0.89655172 0.93103448 1.         1.         0.96428571]
# MLPClassifier 의 정답률 :
#  [0.24137931 0.75862069 0.17857143 0.92857143 0.85714286]
# MultiOutputClassifier 은 없는 놈!
# MultinomialNB 의 정답률 :
#  [0.82758621 0.79310345 0.92857143 0.96428571 0.89285714]
# NearestCentroid 의 정답률 :
#  [0.5862069  0.79310345 0.75       0.75       0.60714286]
# NuSVC 의 정답률 :
#  [0.93103448 0.96551724 0.96428571 0.89285714 0.78571429]
# OneVsOneClassifier 은 없는 놈!
# OneVsRestClassifier 은 없는 놈!
# OutputCodeClassifier 은 없는 놈!
# PassiveAggressiveClassifier 의 정답률 :
#  [0.24137931 0.62068966 0.57142857 0.53571429 0.21428571]
# Perceptron 의 정답률 :
#  [0.51724138 0.27586207 0.42857143 0.71428571 0.53571429]
# QuadraticDiscriminantAnalysis 의 정답률 :
#  [0.96551724 1.         0.92857143 1.         1.        ]
# RadiusNeighborsClassifier 은 없는 놈!
# RandomForestClassifier 의 정답률 :
#  [0.96551724 0.93103448 1.         1.         1.        ]
# RidgeClassifier 의 정답률 :
#  [1.         0.96551724 1.         1.         0.96428571]
# RidgeClassifierCV 의 정답률 :
#  [1.         0.96551724 1.         0.96428571 0.96428571]
# SGDClassifier 의 정답률 :
#  [0.72413793 0.68965517 0.53571429 0.5        0.67857143]
# SVC 의 정답률 :
#  [0.62068966 0.72413793 0.64285714 0.64285714 0.71428571]
# StackingClassifier 은 없는 놈!
# VotingClassifier 은 없는 놈!