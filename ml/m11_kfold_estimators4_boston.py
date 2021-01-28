from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings
from sklearn.datasets import load_boston

warnings.filterwarnings('ignore')

dataset = load_boston()
x= dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=32)
kfold = KFold(n_splits=5, shuffle=True)

allAlgorithms = all_estimators(type_filter='regressor')

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

# ARDRegression 의 정답률 :
#  [0.80296834 0.56060438 0.76259206 0.69661219 0.73729247]
# AdaBoostRegressor 의 정답률 :
#  [0.84374353 0.8306584  0.82397129 0.72524754 0.85670873]
# BaggingRegressor 의 정답률 :
#  [0.89929365 0.92248012 0.69666279 0.88241597 0.82834762]
# BayesianRidge 의 정답률 :
#  [0.6331203  0.59105297 0.75787023 0.8254289  0.7665701 ]
# CCA 의 정답률 :
#  [0.74046794 0.57687222 0.73574963 0.73663719 0.4425112 ]
# DecisionTreeRegressor 의 정답률 :
#  [0.6965099  0.72344309 0.87181956 0.57163323 0.7676811 ]
# DummyRegressor 의 정답률 :
#  [-0.01009341 -0.00048018 -0.00500333 -0.00117675 -0.00600321]
# ElasticNet 의 정답률 :
#  [0.69516109 0.71260149 0.63452983 0.7187279  0.64910788]
# ElasticNetCV 의 정답률 :
#  [0.63532971 0.58319383 0.67606777 0.7641279  0.48650534]
# ExtraTreeRegressor 의 정답률 :
#  [0.80503098 0.57184238 0.71764831 0.67350646 0.85168804]
# ExtraTreesRegressor 의 정답률 :
#  [0.82643549 0.91527856 0.88074823 0.84334061 0.91289537]
# GammaRegressor 의 정답률 :
#  [-0.08413499 -0.03331251 -0.00284381 -0.00065666 -0.00401692]
# GaussianProcessRegressor 의 정답률 :
#  [-6.41783066 -6.8002331  -6.07257825 -6.77695157 -4.50149873]
# GeneralizedLinearRegressor 의 정답률 :
#  [0.64506102 0.61801132 0.71731009 0.70311176 0.64083817]
# GradientBoostingRegressor 의 정답률 :
#  [0.93932064 0.90146751 0.91134029 0.76418965 0.79224709]
# HistGradientBoostingRegressor 의 정답률 :
#  [0.86480113 0.87861728 0.85413086 0.86723488 0.83617871]
# HuberRegressor 의 정답률 :
#  [0.6231679  0.71828401 0.67802987 0.58888892 0.72658808]
# IsotonicRegression 의 정답률 :
#  [nan nan nan nan nan]
# KNeighborsRegressor 의 정답률 :
#  [0.60874358 0.47033349 0.47155006 0.54297625 0.53356863]
# KernelRidge 의 정답률 :
#  [0.83169618 0.77366158 0.57663254 0.70079025 0.56068014]
# Lars 의 정답률 :
#  [0.69375817 0.44865269 0.77035989 0.67772764 0.77271582]
# LarsCV 의 정답률 :
#  [0.78133022 0.55894983 0.63851253 0.77395691 0.69815078]
# Lasso 의 정답률 :
#  [0.65022078 0.61575952 0.56239525 0.69529134 0.70483102]
# LassoCV 의 정답률 :
#  [0.6413044  0.59443327 0.77394895 0.65829342 0.73778322]
# LassoLars 의 정답률 :
#  [-1.10345332e-02 -3.66888157e-02 -4.59013017e-05 -1.44331648e-04
#  -3.20216983e-03]
# LassoLarsCV 의 정답률 :
#  [0.62540064 0.77771084 0.68799563 0.79935216 0.72613709]
# LassoLarsIC 의 정답률 :
#  [0.8006788  0.54299998 0.79773978 0.78121616 0.64599381]
# LinearRegression 의 정답률 :
#  [0.81855702 0.79547953 0.44495042 0.70111964 0.75704123]
# LinearSVR 의 정답률 :
#  [ 0.02790957  0.61771724  0.56691273 -0.79237359  0.24348161]
# MLPRegressor 의 정답률 :
#  [ 0.57081185  0.53733536  0.61683289  0.47250975 -1.26069145]
# MultiOutputRegressor 은 없는 놈!
# MultiTaskElasticNet 의 정답률 :
#  [nan nan nan nan nan]
# MultiTaskElasticNetCV 의 정답률 :
#  [nan nan nan nan nan]
# MultiTaskLasso 의 정답률 :
#  [nan nan nan nan nan]
# MultiTaskLassoCV 의 정답률 :
#  [nan nan nan nan nan]
# NuSVR 의 정답률 :
#  [0.27239554 0.28047844 0.22862499 0.3195825  0.26266364]
# OrthogonalMatchingPursuit 의 정답률 :
#  [0.4739329  0.58041997 0.60661528 0.24171808 0.53470862]
# OrthogonalMatchingPursuitCV 의 정답률 :
#  [0.62276016 0.77808586 0.67326065 0.66030903 0.75811661]
# PLSCanonical 의 정답률 :
#  [-1.76434531 -2.04000368 -2.52735664 -3.89111844 -1.51177682]
# PLSRegression 의 정답률 :
#  [0.66364002 0.79414024 0.68676753 0.5219282  0.78609083]
# PassiveAggressiveRegressor 의 정답률 :
#  [0.02420737 0.24312752 0.29928856 0.01301315 0.24169536]
# PoissonRegressor 의 정답률 :
#  [0.76717728 0.83375523 0.70385739 0.69781872 0.81365693]
# RANSACRegressor 의 정답률 :
#  [ 0.69317538  0.53332108 -1.56117238  0.35689378  0.6006559 ]
# RadiusNeighborsRegressor 은 없는 놈!
# RandomForestRegressor 의 정답률 :
#  [0.82490564 0.91807543 0.88288233 0.87792258 0.8288946 ]
# RegressorChain 은 없는 놈!
# Ridge 의 정답률 :
#  [0.7435135  0.70999948 0.74034091 0.78149245 0.67428888]
# RidgeCV 의 정답률 :
#  [0.73883748 0.71620887 0.74590351 0.66361159 0.77541198]
# SGDRegressor 의 정답률 :
#  [-1.13532225e+25 -7.21836724e+25 -6.19534747e+26 -5.02765405e+26
#  -4.35200156e+25]
# SVR 의 정답률 :
#  [0.40265885 0.16933435 0.2993873  0.2299336  0.07002938]
# StackingRegressor 은 없는 놈!
# TheilSenRegressor 의 정답률 :
#  [0.66389341 0.85654363 0.77531756 0.44768719 0.77495368]
# TransformedTargetRegressor 의 정답률 :
#  [0.72506273 0.67832369 0.67526868 0.81557807 0.70416812]
# TweedieRegressor 의 정답률 :
#  [0.69973007 0.58715619 0.70975045 0.64869744 0.63166415]
# VotingRegressor 은 없는 놈!
# _SigmoidCalibration 의 정답률 :
#  [nan nan nan nan nan]