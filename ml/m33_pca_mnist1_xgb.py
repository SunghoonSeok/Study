import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
print(x.shape) # (70000, 28, 28)
x = x.reshape(70000, 28*28)

# pca = PCA()
# x = pca.fit(x)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print("cumsum :",cumsum)

# d = np.argmax(cumsum >=1)+1
# print("cumsum >=0.95 :", cumsum>=0.95)
# print("d :", d) # d : 674 # 714

pca = PCA(n_components=154)
x2 = pca.fit_transform(x)

x2_train = x2[:60000,:]
x2_test = x2[60000:,:]

model = XGBClassifier(n_jobs=8, use_label_encoder=False)

model.fit(x2_train,y_train, eval_metric='logloss')

#4. 평가, 예측
acc = model.score(x2_test, y_test)

print("acc :", acc)

# acc : 0.9627