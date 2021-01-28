from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings
from sklearn.datasets import load_diabetes

warnings.filterwarnings('ignore')

dataset = load_diabetes()
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
#  [0.39476538 0.40355391 0.56470334 0.62276776 0.39726515]
# AdaBoostRegressor 의 정답률 :
#  [0.4875602  0.54537077 0.30409877 0.4278298  0.46672773]
# BaggingRegressor 의 정답률 :
#  [0.29187755 0.50753689 0.40314272 0.33900429 0.14578829]
# BayesianRidge 의 정답률 :
#  [0.53931283 0.43262889 0.44184123 0.6015416  0.52350675]
# CCA 의 정답률 :
#  [0.63938642 0.54429195 0.35899134 0.4025656  0.48755235]
# DecisionTreeRegressor 의 정답률 :
#  [-0.00915244 -0.06703179 -0.11255589 -0.44202778 -0.12276828]
# DummyRegressor 의 정답률 :
#  [-1.15501356e-03 -4.69283679e-03 -6.99577142e-03 -5.63691036e-06
#  -2.92333952e-03]
# ElasticNet 의 정답률 :
#  [-0.13903859  0.00431393 -0.03356674  0.00726751 -0.00198272]
# ElasticNetCV 의 정답률 :
#  [0.41580368 0.43426717 0.48247786 0.4190275  0.44671911]
# ExtraTreeRegressor 의 정답률 :
#  [-0.04054267 -0.08666175 -0.03873468  0.05796968 -0.36952735]
# ExtraTreesRegressor 의 정답률 :
#  [0.52705067 0.31183223 0.36260166 0.41043987 0.41780798]
# GammaRegressor 의 정답률 :
#  [-0.03391227 -0.02324726 -0.02287651 -0.01419988  0.00386369]
# GaussianProcessRegressor 의 정답률 :
#  [-10.2383724  -13.0493915  -10.01183696 -13.70222721 -27.50581157]
# GeneralizedLinearRegressor 의 정답률 :
#  [ 0.00466284  0.00560445  0.00620741 -0.0092322  -0.01422451]
# GradientBoostingRegressor 의 정답률 :
#  [0.30366909 0.35448659 0.42637328 0.35121281 0.45552118]
# HistGradientBoostingRegressor 의 정답률 :
#  [0.59739785 0.3424249  0.21562927 0.28398424 0.48046323]
# HuberRegressor 의 정답률 :
#  [0.437546   0.59070502 0.35366698 0.453517   0.60813864]
# IsotonicRegression 의 정답률 :
#  [nan nan nan nan nan]
# KNeighborsRegressor 의 정답률 :
#  [0.30437861 0.38316205 0.31222328 0.56647999 0.37330588]
# KernelRidge 의 정답률 :
#  [-4.26445218 -4.19505326 -3.72060495 -3.43263991 -2.56130996]
# Lars 의 정답률 :
#  [0.45995192 0.48247025 0.53184782 0.1201273  0.34799156]
# LarsCV 의 정답률 :
#  [0.36487829 0.57338112 0.50156876 0.53087647 0.52624192]
# Lasso 의 정답률 :
#  [0.34799196 0.31235827 0.37234285 0.3674886  0.38321902]
# LassoCV 의 정답률 :
#  [0.43615305 0.53930151 0.62736445 0.44156281 0.44444879]
# LassoLars 의 정답률 :
#  [0.37784673 0.40005588 0.38075496 0.39358878 0.38817252]
# LassoLarsCV 의 정답률 :
#  [0.54904003 0.45173682 0.56720392 0.32396279 0.51497393]
# LassoLarsIC 의 정답률 :
#  [0.55230169 0.54072576 0.42395365 0.43391512 0.51720363]
# LinearRegression 의 정답률 :
#  [0.37353817 0.63559413 0.48122909 0.44942734 0.54340592]
# LinearSVR 의 정답률 :
#  [-0.54630405 -0.52669859 -0.30655719 -0.48955247 -0.59998587]
# MLPRegressor 의 정답률 :
#  [-2.76520102 -2.10562457 -4.32598408 -3.20069817 -3.1574601 ]
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
#  [0.13242892 0.13522438 0.13077555 0.14848071 0.10121992]
# OrthogonalMatchingPursuit 의 정답률 :
#  [0.24285166 0.33964092 0.28683747 0.30982417 0.29733603]
# OrthogonalMatchingPursuitCV 의 정답률 :
#  [0.50403933 0.2990644  0.53920308 0.51033292 0.51353606]
# PLSCanonical 의 정답률 :
#  [-1.05752849 -0.9783461  -0.35847566 -3.00080287 -1.03377314]
# PLSRegression 의 정답률 :
#  [0.6715951  0.53632304 0.47414344 0.31465051 0.40324681]
# PassiveAggressiveRegressor 의 정답률 :
#  [0.51718745 0.54991434 0.28784559 0.39932822 0.49900474]
# PoissonRegressor 의 정답률 :
#  [0.36262921 0.34361215 0.27217482 0.32436452 0.35450838]
# RANSACRegressor 의 정답률 :
#  [ 0.13111839 -1.45110718 -0.10134669  0.30786172 -0.09329172]
# RadiusNeighborsRegressor 의 정답률 :
#  [-0.01482575 -0.00995324 -0.009256   -0.00017874 -0.00063373]
# RandomForestRegressor 의 정답률 :
#  [0.38605782 0.49624628 0.5620633  0.20715593 0.33091614]
# RegressorChain 은 없는 놈!
# Ridge 의 정답률 :
#  [0.43635436 0.43454286 0.41003887 0.39743342 0.34152998]
# RidgeCV 의 정답률 :
#  [0.55421719 0.50451919 0.54270669 0.36804126 0.47869897]
# SGDRegressor 의 정답률 :
#  [0.47383488 0.41272803 0.33365025 0.37690246 0.40132747]
# SVR 의 정답률 :
#  [0.15732123 0.15131903 0.0291303  0.11932408 0.097624  ]
# StackingRegressor 은 없는 놈!
# TheilSenRegressor 의 정답률 :
#  [0.55416763 0.41743093 0.46119746 0.50751066 0.4019914 ]
# TransformedTargetRegressor 의 정답률 :
#  [0.50737828 0.60656149 0.40506308 0.49083156 0.41019029]
# TweedieRegressor 의 정답률 :
#  [-0.00521445 -0.04976968 -0.01503763  0.00314728 -0.07764591]
# VotingRegressor 은 없는 놈!
# _SigmoidCalibration 의 정답률 :
#  [nan nan nan nan nan]